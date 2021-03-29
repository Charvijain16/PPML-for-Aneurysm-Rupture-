
import json
import os
import pickle
import tarfile

import numpy as np
import tensorflow as tf
from tensorpack import (
    BatchData, BatchNorm, Dropout, FullyConnected, ModelSaver,
    PredictConfig, QueueInput, SaverRestore, SimpleDatasetPredictor)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized
from tensorpack.utils import logger
from tensorpack.graph_builder.model_desc import InputDesc
from tensorpack.graph_builder import ModelDescBase

from data import RandomZData, TGANDataFlow
from data import Preprocessor
from trainer import GANTrainer
from tensorflow_privacy.privacy.analysis import privacy_ledger, compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer

TUNABLE_VARIABLES = {
    'batch_size': [10],
    'z_dim': [10, 20, 30, 50, 100, 200],
    'num_gen_rnn': [50, 100, 200, 300],
    'num_gen_feature': [50, 100, 200, 300],
    'num_dis_layers': [1, 2, 3, 4, 5],
    'num_dis_hidden': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.02, 0.5, 0.1, 0.2],
    'noise': [0.05, 0.1, 0.2, 0.3],
    'max_epoch': [10, 15, 20, 30, 40],
    'microbatches': [1, 5, 10],
    'noise_multiplier': [3],
    'l2_norm_clip': [0.01, 0.05, 0.02, 0.1, 0.2, 0.5, 1.1, 2]

}

'''
TUNABLE_VARIABLES = {
    'batch_size': [50, 100, 200],
    'z_dim': [50, 100, 200, 400],
    'num_gen_rnn': [100, 200, 300, 400, 500, 600],
    'num_gen_feature': [100, 200, 300, 400, 500, 600],
    'num_dis_layers': [1, 2, 3, 4, 5],
    'num_dis_hidden': [100, 200, 300, 400, 500],
    'learning_rate': [0.0002, 0.0005, 0.001],
    'noise': [0.05, 0.1, 0.2, 0.3],
    'max_epoch': [10, 15, 20],
}
'''



class GraphBuilder(ModelDescBase):

    def __init__(
            self,
            metadata,
            batch_size=200,
            z_dim=200,
            noise=0.2,
            l2norm=0.00001,
            learning_rate=0.001,
            num_gen_rnn=100,
            num_gen_feature=100,
            num_dis_layers=1,
            num_dis_hidden=100,
            optimizer='GradientDescentOptimizer',
            training=True,
            disc_optimizer='DPGradientDescentGaussianOptimizer',
            microbatches=200,
            noise_multiplier=0.01,
            l2_norm_clip=0.000002,
            max_epoch=10
    ):
        self.epochs = max_epoch
        self.disc_optimizer = disc_optimizer
        self.microbatches = microbatches
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.metadata = metadata
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_feature = num_gen_feature
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.optimizer = optimizer
        self.training = training

    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)

        if not (self.g_vars or self.d_vars):
            raise ValueError('There are no variables defined in some of the given scopes')

    def build_losses(self, logits_real, logits_fake, extra_g=0, l2_norm=0.00001):

        with tf.name_scope("GAN_loss"):
            score_real = tf.sigmoid(logits_real)
            score_fake = tf.sigmoid(logits_fake)

            tf.summary.histogram('score-real', score_real)
            tf.summary.histogram('score-fake', score_fake)
            with tf.name_scope("discrim"):
                d_loss_pos = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=logits_real,
                        labels=tf.ones_like(logits_real)) * 0.7 + tf.random_uniform(
                        tf.shape(logits_real),
                        maxval=0.3
                    ),
                    name='loss_real'
                )

                d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss_fake')

                d_pos_acc = tf.reduce_mean(
                    tf.cast(score_real > 0.5, tf.float32), name='accuracy_real')

                d_neg_acc = tf.reduce_mean(
                    tf.cast(score_fake < 0.5, tf.float32), name='accuracy_fake')

                d_loss = 0.5 * d_loss_pos + 0.5 * d_loss_neg + \
                         tf.contrib.layers.apply_regularization(
                             tf.contrib.layers.l2_regularizer(l2_norm),
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim"))

                self.d_loss = tf.identity(d_loss, name='loss')

            with tf.name_scope("gen"):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake))) + \
                         tf.contrib.layers.apply_regularization(
                             tf.contrib.layers.l2_regularizer(l2_norm),
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen'))

                g_loss = tf.identity(g_loss, name='loss')
                extra_g = tf.identity(extra_g, name='klloss')
                self.g_loss = tf.identity(g_loss + extra_g, name='final-g-loss')

            add_moving_summary(
                g_loss, extra_g, self.g_loss, self.d_loss, d_pos_acc, d_neg_acc, decay=0.)

    @memoized
    def get_optimizer(self):
        return self._get_optimizer()

    @memoized
    def get_disc_optimizer(self):
        return self._get_disc_optimizer()

    def inputs(self):
        inputs = []
        for col_id, col_info in enumerate(self.metadata['details']):
            if col_info['type'] == 'value':
                gaussian_components = col_info['n']
                inputs.append(
                    InputDesc(tf.float32, (self.batch_size, 1), 'input%02dvalue' % col_id))

                inputs.append(
                    InputDesc(
                        tf.float32,
                        (self.batch_size, gaussian_components),
                        'input%02dcluster' % col_id
                    )
                )

            elif col_info['type'] == 'category':
                inputs.append(InputDesc(tf.int32, (self.batch_size, 1), 'input%02d' % col_id))

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`values`. Instead it was {}.".format(col_id, col_info['type'])
                )

        return inputs

    def generator(self, z):
        with tf.variable_scope('LSTM'):
            cell = tf.nn.rnn_cell.LSTMCell(self.num_gen_rnn)

            state = cell.zero_state(self.batch_size, dtype='float32')
            attention = tf.zeros(
                shape=(self.batch_size, self.num_gen_rnn), dtype='float32')
            input = tf.get_variable(name='go', shape=(1, self.num_gen_feature))  # <GO>
            input = tf.tile(input, [self.batch_size, 1])
            input = tf.concat([input, z], axis=1)

            ptr = 0
            outputs = []
            states = []
            for col_id, col_info in enumerate(self.metadata['details']):
                if col_info['type'] == 'value':
                    output, state = cell(tf.concat([input, attention], axis=1), state)
                    states.append(state[1])

                    gaussian_components = col_info['n']
                    with tf.variable_scope("%02d" % ptr):
                        h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)
                        outputs.append(FullyConnected('FC2', h, 1, nl=tf.tanh))
                        input = tf.concat([h, z], axis=1)
                        attw = tf.get_variable("attw", shape=(len(states), 1, 1))
                        attw = tf.nn.softmax(attw, axis=0)
                        attention = tf.reduce_sum(tf.stack(states, axis=0) * attw, axis=0)

                    ptr += 1

                    output, state = cell(tf.concat([input, attention], axis=1), state)
                    states.append(state[1])
                    with tf.variable_scope("%02d" % ptr):
                        h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)
                        w = FullyConnected('FC2', h, gaussian_components, nl=tf.nn.softmax)
                        outputs.append(w)
                        input = FullyConnected('FC3', w, self.num_gen_feature, nl=tf.identity)
                        input = tf.concat([input, z], axis=1)
                        attw = tf.get_variable("attw", shape=(len(states), 1, 1))
                        attw = tf.nn.softmax(attw, axis=0)
                        attention = tf.reduce_sum(tf.stack(states, axis=0) * attw, axis=0)

                    ptr += 1

                elif col_info['type'] == 'category':
                    output, state = cell(tf.concat([input, attention], axis=1), state)
                    states.append(state[1])
                    with tf.variable_scope("%02d" % ptr):
                        h = FullyConnected('FC', output, self.num_gen_feature, nl=tf.tanh)
                        w = FullyConnected('FC2', h, col_info['n'], nl=tf.nn.softmax)
                        outputs.append(w)
                        one_hot = tf.one_hot(tf.argmax(w, axis=1), col_info['n'])
                        input = FullyConnected(
                            'FC3', one_hot, self.num_gen_feature, nl=tf.identity)
                        input = tf.concat([input, z], axis=1)
                        attw = tf.get_variable("attw", shape=(len(states), 1, 1))
                        attw = tf.nn.softmax(attw, axis=0)
                        attention = tf.reduce_sum(tf.stack(states, axis=0) * attw, axis=0)

                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`values`. Instead it was {}.".format(col_id, col_info['type'])
                    )

        return outputs

    @staticmethod
    def batch_diversity(l, n_kernel=10, kernel_dim=10):
        M = FullyConnected('fc_diversity', l, n_kernel * kernel_dim, nl=tf.identity)
        M = tf.reshape(M, [-1, n_kernel, kernel_dim])
        M1 = tf.reshape(M, [-1, 1, n_kernel, kernel_dim])
        M2 = tf.reshape(M, [1, -1, n_kernel, kernel_dim])
        diff = tf.exp(-tf.reduce_sum(tf.abs(M1 - M2), axis=3))
        return tf.reduce_sum(diff, axis=0)

    @auto_reuse_variable_scope
    def discriminator(self, vecs):

        logits = tf.concat(vecs, axis=1)
        for i in range(self.num_dis_layers):
            with tf.variable_scope('dis_fc{}'.format(i)):
                if i == 0:
                    logits = FullyConnected(
                        'fc', logits, self.num_dis_hidden, nl=tf.identity,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )

                else:
                    logits = FullyConnected('fc', logits, self.num_dis_hidden, nl=tf.identity)

                logits = tf.concat([logits, self.batch_diversity(logits)], axis=1)
                logits = BatchNorm('bn', logits, center=True, scale=False)
                logits = Dropout(logits)
                logits = tf.nn.leaky_relu(logits)

        return FullyConnected('dis_fc_top', logits, 1, nl=tf.identity)

    @staticmethod
    def compute_kl(real, pred):

        return tf.reduce_sum((tf.log(pred + 1e-4) - tf.log(real + 1e-4)) * pred)

    def build_graph(self, *inputs):

        z = tf.random_normal(
            [self.batch_size, self.z_dim], name='z_train')

        z = tf.placeholder_with_default(z, [None, self.z_dim], name='z')

        with tf.variable_scope('gen'):
            vecs_gen = self.generator(z)

            vecs_denorm = []
            ptr = 0
            for col_id, col_info in enumerate(self.metadata['details']):
                if col_info['type'] == 'category':
                    t = tf.argmax(vecs_gen[ptr], axis=1)
                    t = tf.cast(tf.reshape(t, [-1, 1]), 'float32')
                    vecs_denorm.append(t)
                    ptr += 1

                elif col_info['type'] == 'value':
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`values`. Instead it was {}.".format(col_id, col_info['type'])
                    )

            tf.identity(tf.concat(vecs_denorm, axis=1), name='gen')

        vecs_pos = []
        ptr = 0
        for col_id, col_info in enumerate(self.metadata['details']):
            if col_info['type'] == 'category':
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])
                noise_input = one_hot

                if self.training:
                    noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=self.noise)
                    noise_input = (one_hot + noise) / tf.reduce_sum(
                        one_hot + noise, keepdims=True, axis=1)

                vecs_pos.append(noise_input)
                ptr += 1

            elif col_info['type'] == 'value':
                vecs_pos.append(inputs[ptr])
                ptr += 1
                vecs_pos.append(inputs[ptr])
                ptr += 1

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`values`. Instead it was {}.".format(col_id, col_info['type'])
                )

        KL = 0.
        ptr = 0
        if self.training:
            for col_id, col_info in enumerate(self.metadata['details']):
                if col_info['type'] == 'category':
                    dist = tf.reduce_sum(vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(dist)

                    real = tf.reduce_sum(vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(real)
                    KL += self.compute_kl(real, dist)
                    ptr += 1

                elif col_info['type'] == 'value':
                    ptr += 1
                    dist = tf.reduce_sum(vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(dist)
                    real = tf.reduce_sum(vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(real)
                    KL += self.compute_kl(real, dist)

                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`values`. Instead it was {}.".format(col_id, col_info['type'])
                    )

        with tf.variable_scope('discrim'):
            discrim_pos = self.discriminator(vecs_pos)
            discrim_neg = self.discriminator(vecs_gen)

        self.build_losses(discrim_pos, discrim_neg, extra_g=KL, l2_norm=self.l2norm)
        self.collect_variables()

    def _get_optimizer(self):
        if self.optimizer == 'AdamOptimizer':
            return tf.train.AdamOptimizer(self.learning_rate, 0.5)

        elif self.optimizer == 'AdadeltaOptimizer':
            return tf.train.AdadeltaOptimizer(self.learning_rate, 0.95)

        else:
            return tf.train.GradientDescentOptimizer(self.learning_rate)

    def _get_disc_optimizer(self, NB_TRAIN=447):
        if self.disc_optimizer == 'DPGradientDescentGaussianOptimizer':
            ledger = privacy_ledger.PrivacyLedger(
                population_size=NB_TRAIN,
                selection_probability=(self.batch_size / NB_TRAIN))

            LR_DISC = tf.compat.v1.train.polynomial_decay(learning_rate=0.150,
                                                          global_step=tf.compat.v1.train.get_or_create_global_step(),
                                                          decay_steps=10000,
                                                          end_learning_rate=0.052,
                                                          power=1)
            # return tf.train.GradientDescentOptimizer(self.learning_rate)
            return dp_optimizer.DPGradientDescentGaussianOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=1,
                ledger=ledger,
                learning_rate=LR_DISC,
            )

    def compute_epsilon(self):

        if self.noise_multiplier == 0.0:
            return float('inf')
        noise_multiplier = self.noise_multiplier
        sampling_probability = self.batch_size / 447
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        epsilon = []
        for epoch in range(1, self.epochs + 1):
            steps = epoch * 447 // self.batch_size
            rdp = compute_rdp(q=sampling_probability,
                              noise_multiplier=noise_multiplier,
                              steps=steps,
                              orders=orders)
            epsilon.append(get_privacy_spent(orders, rdp, target_delta=0.002)[0])

        return epsilon


class TGANModel:
    def __init__(
            self, continuous_columns, output='output', gpu=None, max_epoch=5, steps_per_epoch=10000,
            save_checkpoints=True, restore_session=True, batch_size=200, z_dim=200, noise=0.2,
            l2norm=0.00001, learning_rate=0.001, num_gen_rnn=100, num_gen_feature=100,
            num_dis_layers=1, num_dis_hidden=100, optimizer='GradientDescentOptimizer',
            disc_optimizer='DPGradientDescentGaussianOptimizer',
            microbatches=200,
            noise_multiplier=0.01,
            l2_norm_clip=0.000002
    ):
        # Output
        self.continuous_columns = continuous_columns
        self.log_dir = os.path.join(output, 'logs')
        self.model_dir = os.path.join(output, 'model')
        self.output = output

        # Training params
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self.save_checkpoints = save_checkpoints
        self.restore_session = restore_session

        # Model params
        self.model = None
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_feature = num_gen_feature
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.microbatches = microbatches
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip

        if gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        self.gpu = gpu

    def get_model(self, training=True):
        return GraphBuilder(
            metadata=self.metadata,
            batch_size=self.batch_size,
            z_dim=self.z_dim,
            noise=self.noise,
            l2norm=self.l2norm,
            learning_rate=self.learning_rate,
            num_gen_rnn=self.num_gen_rnn,
            num_gen_feature=self.num_gen_feature,
            num_dis_layers=self.num_dis_layers,
            num_dis_hidden=self.num_dis_hidden,
            optimizer=self.optimizer,
            training=training,
            disc_optimizer='DPGradientDescentGaussianOptimizer',
            microbatches=200,
            noise_multiplier=0.01,
            l2_norm_clip=0.000002,
            max_epoch=10
        )

    def prepare_sampling(self):
        if self.model is None:
            self.model = self.get_model(training=False)

        else:
            self.model.training = False

        predict_config = PredictConfig(
            session_init=SaverRestore(self.restore_path),
            model=self.model,
            input_names=['z'],
            output_names=['gen/gen', 'z'],
        )

        self.simple_dataset_predictor = SimpleDatasetPredictor(
            predict_config,
            RandomZData((self.batch_size, self.z_dim))
        )

    def fit(self, data):
        self.preprocessor = Preprocessor(continuous_columns=self.continuous_columns)
        data = self.preprocessor.fit_transform(data)
        self.metadata = self.preprocessor.metadata
        dataflow = TGANDataFlow(data, self.metadata)
        batch_data = BatchData(dataflow, self.batch_size)
        input_queue = QueueInput(batch_data)

        self.model = self.get_model(training=True)

        trainer = GANTrainer(
            model=self.model,
            input_queue=input_queue,
        )

        self.restore_path = os.path.join(self.model_dir, 'checkpoint')

        if os.path.isfile(self.restore_path) and self.restore_session:
            session_init = SaverRestore(self.restore_path)
            with open(os.path.join(self.log_dir, 'stats.json')) as f:
                starting_epoch = json.load(f)[-1]['epoch_num'] + 1

        else:
            session_init = None
            starting_epoch = 1

        action = 'k' if self.restore_session else None
        logger.set_logger_dir(self.log_dir, action=action)

        callbacks = []
        if self.save_checkpoints:
            callbacks.append(ModelSaver(checkpoint_dir=self.model_dir))

        trainer.train_with_defaults(
            callbacks=callbacks,
            steps_per_epoch=self.steps_per_epoch,
            max_epoch=self.max_epoch,
            session_init=session_init,
            starting_epoch=starting_epoch
        )

        self.prepare_sampling()

    def sample(self, num_samples):

        max_iters = (num_samples // self.batch_size)

        results = []
        for idx, o in enumerate(self.simple_dataset_predictor.get_result()):
            results.append(o[0])
            if idx + 1 == max_iters:
                break

        results = np.concatenate(results, axis=0)

        ptr = 0
        features = {}
        for col_id, col_info in enumerate(self.metadata['details']):
            if col_info['type'] == 'category':
                features['f%02d' % col_id] = results[:, ptr:ptr + 1]
                ptr += 1

            elif col_info['type'] == 'value':
                gaussian_components = col_info['n']
                val = results[:, ptr:ptr + 1]
                ptr += 1
                pro = results[:, ptr:ptr + gaussian_components]
                ptr += gaussian_components
                features['f%02d' % col_id] = np.concatenate([val, pro], axis=1)

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`values`. Instead it was {}.".format(col_id, col_info['type'])
                )

        return self.preprocessor.reverse_transform(features)[:num_samples].copy()

    def tar_folder(self, tar_name):
        with tarfile.open(tar_name, 'w:gz') as tar_handle:
            for root, dirs, files in os.walk(self.output):
                for file_ in files:
                    tar_handle.add(os.path.join(root, file_))

            tar_handle.close()

    @classmethod
    def load(cls, path):
        with tarfile.open(path, 'r:gz') as tar_handle:
            destination_dir = os.path.dirname(tar_handle.getmembers()[0].name)
            tar_handle.extractall()

        with open('{}/TGANModel'.format(destination_dir), 'rb') as f:
            instance = pickle.load(f)

        instance.prepare_sampling()
        return instance

    def save(self, path, force=False):
        if os.path.exists(path) and not force:
            logger.info('The indicated path already exists. Use `force=True` to overwrite.')
            return

        base_path = os.path.dirname(path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model = self.model
        dataset_predictor = self.simple_dataset_predictor

        self.model = None
        self.simple_dataset_predictor = None

        with open('{}/TGANModel'.format(self.output), 'wb') as f:
            pickle.dump(self, f)

        self.model = model
        self.simple_dataset_predictor = dataset_predictor

        self.tar_folder(path)

        logger.info('Model saved successfully.')
