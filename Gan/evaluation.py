import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def evaluate_classification_on_newdataset(
        train_data, test_data):
    train_data.replace(np.nan, 0)
    Y_train = train_data.label.values
    X_train = train_data.drop("label", axis=1).values
    Y_test = test_data.label.values
    X_test = test_data.drop("label", axis=1).values

    model = RandomForestClassifier(n_estimators=3, random_state=30, max_depth=1, min_samples_leaf=5,
                                   min_samples_split=10)

    model.fit(X_train, Y_train)
    prediction_test = model.predict(X_test)
    acc = metrics.f1_score(Y_test, prediction_test)
    print("Accuracy", acc)
    return acc


def evaluate_classification_on_originaldataset(
        train_data, test_data
):
    test_data.replace(np.nan, 0)

    Y_train = train_data.label.values
    X_train = train_data.drop("label", axis=1).values
    Y_test = test_data.label.values
    X_test = test_data.drop("label", axis=1).values

    model = RandomForestClassifier(n_estimators=3, random_state=30, max_depth=1, min_samples_leaf=5,
                                   min_samples_split=10)

    model.fit(X_train, Y_train)
    prediction_test = model.predict(X_test)
    acc = metrics.f1_score(Y_test, prediction_test)
    print("Accuracy", acc)
    return acc
