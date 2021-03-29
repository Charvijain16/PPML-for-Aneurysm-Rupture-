import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorpack.utils import logger

from evaluation import evaluate_classification_on_newdataset, evaluate_classification_on_originaldataset
from model import TUNABLE_VARIABLES, TGANModel
from table_evaluator import load_data, TableEvaluator


def comparison(location_synthteic_data):
    df1 = pd.read_csv("../Preprocessing/preprocessed_data1.csv", encoding='ISO-8859-3')
    model_0 = pd.read_csv(location_synthteic_data)
    print(len(df1), len(model_0))
    table_evaluator = TableEvaluator(df1, model_0)
    table_evaluator.visual_evaluation()


def prepare_hyperparameter_search(steps_per_epoch, num_random_search):
    model_kwargs = []
    basic_kwargs = {
        'steps_per_epoch': steps_per_epoch,
    }

    for i in range(num_random_search):
        kwargs = {name: np.random.choice(choices) for name, choices in TUNABLE_VARIABLES.items()}
        kwargs.update(basic_kwargs)
        model_kwargs.append(kwargs)

    return model_kwargs


def fit_score_model(
        name, model_kwargs, train_data, test_data, continuous_columns,
        sample_rows, store_samples
):

    for index, kwargs in enumerate(model_kwargs):
        logger.info('Training TGAN Model %d/%d', index + 1, len(model_kwargs))

        tf.reset_default_graph()
        base_dir = os.path.join('experiments', name)
        output = os.path.join(base_dir, 'model_{}'.format(index))
        model = TGANModel(continuous_columns, output=output, **kwargs)
        model.fit(train_data)
        sampled_data = model.sample(sample_rows)
        sampled_data.columns = ['doppelt', 'GeschlechtNum', 'Alter',
                                'AlterGebärf', 'Altersklassen',
                                'BMI', 'BMIKlassen', 'artHyperCodiert',
                                'DiabMellCodiert',
                                'BlutverdünnerCodiert', 'NikotinCodiert',
                                'PHASES', 'AnzahlAns',
                                'TrägZusCodiert', 'TrägSeitCodiert',
                                'SABAnsLänge', 'SABAnsBreite',
                                'LängeGruppiert', 'LängePhases', 'FormCodiert',
                                'cerebrAnoCodiert',
                                'VglOffenInterv', 'TherCodiert',
                                'Zustand30Tage', 'Zustand1Jahr',
                                'btnlCodiert', 'GOSlängstmöglich',
                                'WiedereinstCodiert',
                                'GrössnachInterCodiert', 'ZeitzumReEinstrom',
                                'ZeitinIntervallen',
                                'ZweiteingriffeAne', 'ZweiteingrCodiert',
                                'RuptCodiert', 'KomplCodiert',
                                'filter_$', 'GrössWährCodiert',
                                'YearsfromBeginning', 'label']

        if store_samples:
            dir_name = os.path.join(base_dir, 'data')
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            file_name = os.path.join(dir_name, 'model_{}.csv'.format(index))
            sampled_data.to_csv(file_name, index=False, header=True)
            comparison(file_name)

        score = evaluate_classification_on_originaldataset(sampled_data, test_data)
        model_kwargs[index]['score'] = score

        score_on_new = evaluate_classification_on_newdataset(sampled_data, test_data)
        score_on_original = evaluate_classification_on_originaldataset(train_data, sampled_data)
        print("score_on_new:", score_on_new, "score_on_original:", score_on_original)
        model_kwargs[index]['score_on_original', 'score_on_new'] = score_on_original, score_on_new
        print(model_kwargs)
    return model_kwargs


def run_experiment(
        name, steps_per_epoch, sample_rows, train_csv, continuous_cols,
        num_random_search, store_samples=True, force=False
):
    if os.path.isdir(name):
        if force:
            logger.info('Folder "{}" exists, and force=True. Deleting folder.'.format(name))
            os.rmdir(name)

        else:
            raise ValueError(
                'Folder "{}" already exist. Please, use force=True to force deletion '
                'or use a different name.'.format(name))

    data = pd.read_csv(train_csv, encoding='ISO-8859-3')
    train_data, test_data = train_test_split(data, train_size=0.8, random_state=2)
    model_kwargs = prepare_hyperparameter_search(steps_per_epoch, num_random_search)

    return fit_score_model(
        name, model_kwargs, train_data, test_data,
        continuous_cols, sample_rows, store_samples
    )
