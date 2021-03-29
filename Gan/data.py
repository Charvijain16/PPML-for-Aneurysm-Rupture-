import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from tensorpack import DataFlow, RNGDataFlow

def check_metadata(metadata):
    message = 'The given metadata contains unsupported types.'
    assert all([item['type'] in ['category', 'value'] for item in metadata['details']]), message
def check_inputs(function):
    def decorated(self, data, *args, **kwargs):
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] == 1):
            raise ValueError('The argument `data` must be a numpy.ndarray with shape (n, 1).')

        return function(self, data, *args, **kwargs)

    decorated.__doc__ = function.__doc__
    return decorated

class RandomZData(DataFlow):
    def __init__(self, shape):
        """Initialize object."""
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        """Yield random normal vectors of shape :attr:`shape`."""
        while True:
            yield [np.random.normal(0, 1, size=self.shape)]

    def __iter__(self):
        """Return data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.shape[0]
class MultiModalNumberTransformer:
    def __init__(self, num_modes=5):
        """Initialize instance."""
        self.num_modes = num_modes

    @check_inputs
    def transform(self, data):

        model = GaussianMixture(self.num_modes)
        model.fit(data)

        means = model.means_.reshape((1, self.num_modes))
        stds = np.sqrt(model.covariances_).reshape((1, self.num_modes))

        features = (data - means) / (2 * stds)
        probs = model.predict_proba(data)
        argmax = np.argmax(probs, axis=1)
        idx = np.arange(len(features))
        features = features[idx, argmax].reshape([-1, 1])

        features = np.clip(features, -0.99, 0.99)

        return features, probs, list(means.flat), list(stds.flat)

    @staticmethod
    def inverse_transform(data, info):
        """Reverse the clustering of values.

        Args:
            data(numpy.ndarray): Transformed data to restore.
            info(dict): Metadata.

        Returns:
           numpy.ndarray: Values in the original space.

        """
        features = data[:, 0]
        probs = data[:, 1:]
        p_argmax = np.argmax(probs, axis=1)

        mean = np.asarray(info['means'])
        std = np.asarray(info['stds'])

        select_mean = mean[p_argmax]
        select_std = std[p_argmax]

        return features * 2 * select_std + select_mean

class Preprocessor:


    def __init__(self, continuous_columns=None, metadata=None):
        """Initialize object, set arguments as attributes, initialize transformers."""
        if continuous_columns is None:
            continuous_columns = []

        self.continuous_columns = continuous_columns
        self.metadata = metadata
        self.continous_transformer = MultiModalNumberTransformer()
        self.categorical_transformer = LabelEncoder()
        self.columns = None

    def fit_transform(self, data, fitting=True):
        """Transform human-readable data into TGAN numerical features.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        num_cols = data.shape[1]
        self.columns = data.columns
        data.columns = list(range(num_cols))

        transformed_data = {}
        details = []

        for i in data.columns:
            if i in self.continuous_columns:
                column_data = data[i].values.reshape([-1, 1])
                features, probs, means, stds = self.continous_transformer.transform(column_data)
                transformed_data['f%02d' % i] = np.concatenate((features, probs), axis=1)

                if fitting:
                    details.append({
                        "type": "value",
                        "means": means,
                        "stds": stds,
                        "n": 5
                    })

            else:
                column_data = data[i].astype(str).values
                features = self.categorical_transformer.fit_transform(column_data)
                transformed_data['f%02d' % i] = features.reshape([-1, 1])

                if fitting:
                    mapping = self.categorical_transformer.classes_
                    details.append({
                        "type": "category",
                        "mapping": mapping,
                        "n": mapping.shape[0],
                    })

        if fitting:
            metadata = {
                "num_features": num_cols,
                "details": details
            }
            check_metadata(metadata)
            self.metadata = metadata

        return transformed_data

    def transform(self, data):
        """Transform the given dataframe without generating new metadata.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        return self.fit_transform(data, fitting=False)

    def fit(self, data):
        """Initialize the internal state of the object using :attr:`data`.

        Args:
            data(pandas.DataFrame): Data to fit the object.

        """
        self.fit_transform(data)

    def reverse_transform(self, data):
        """Transform TGAN numerical features back into human-readable data.

        Args:
            data(pandas.DataFrame): Data to transform.
            fitting(bool): Whether or not to update self.metadata.

        Returns:
            pandas.DataFrame: Model features

        """
        table = []

        for i in range(self.metadata['num_features']):
            column_data = data['f%02d' % i]
            column_metadata = self.metadata['details'][i]

            if column_metadata['type'] == 'value':
                column = self.continous_transformer.inverse_transform(column_data, column_metadata)

            if column_metadata['type'] == 'category':
                self.categorical_transformer.classes_ = column_metadata['mapping']
                column = self.categorical_transformer.inverse_transform(
                    column_data.ravel().astype(np.int32))

            table.append(column)

        result = pd.DataFrame(dict(enumerate(table)))
        result.columns = self.columns
        return result



class TGANDataFlow(RNGDataFlow):

    def __init__(self, data, metadata, shuffle=True):
        """Initialize object.

        Args:
            filename(str): Path to the json file containing the metadata.
            shuffle(bool): Wheter or not to shuffle the data.

        Raises:
            ValueError: If any column_info['type'] is not supported

        """
        self.shuffle = shuffle
        if self.shuffle:
            self.reset_state()

        self.metadata = metadata
        self.num_features = self.metadata['num_features']

        self.data = []
        self.distribution = []
        for column_id, column_info in enumerate(self.metadata['details']):
            if column_info['type'] == 'value':
                col_data = data['f%02d' % column_id]
                value = col_data[:, :1]
                cluster = col_data[:, 1:]
                self.data.append(value)
                self.data.append(cluster)

            elif column_info['type'] == 'category':
                col_data = np.asarray(data['f%02d' % column_id], dtype='int32')
                self.data.append(col_data)

            else:
                raise ValueError(
                    "column_info['type'] must be either 'category' or 'value'."
                    "Instead it was '{}'.".format(column_info['type'])
                )

        self.data = list(zip(*self.data))

    def size(self):
        """Return the number of rows in data.

        Returns:
            int: Number of rows in :attr:`data`.

        """
        return len(self.data)

    def get_data(self):
        """Yield the rows from :attr:`data`.

        Yields:
            tuple: Row of data.

        """
        idxs = np.arange(len(self.data))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for k in idxs:
            yield self.data[k]

    def __iter__(self):
        """Iterate over self.data."""
        return self.get_data()

    def __len__(self):
        """Length of batches."""
        return self.size()
