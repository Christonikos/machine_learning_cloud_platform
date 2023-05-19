'''
This is the testing and logging script for the
"Predict Customer Churn" Project of the MLDevOps Nanodegree
'''
import os
import logging
import churn_library_update as clu
import numpy as np
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
        '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    perform_eda(df)
    path = './images/eda'
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: It does not appear that the you "
                        "are correctly saving images to the eda folder.")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        df = encoder_helper(df, cat_columns, 'Churn')

        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")

    except NameError as err:
        logging.error("Name is not defined")
        raise err

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe appears to be missing the "
            "transformed categorical columns")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return X_train, X_test, y_train, y_test


def test_X_scaler(X_scaler, X_train):
    '''
    test X_scaler
    '''
    X_scaled = X_scaler(X_train)

    try:
        assert isinstance(X_scaled, np.ndarray)
        logging.info('Testing X_scaler: SUCCESS')
    except AssertionError as err:
        logging.warning(
            'Testing X_scaler: Scaled X_train is of incorrect type')
        raise err
    return X_scaled

def test_scaled_to_df(scaled_to_df, X_scaled):
    '''
    test scaled_to_df
    '''
    X_scaled_to_df = scaled_to_df(X_scaled)

    try:

        assert isinstance(X_scaled_to_df, pd.DataFrame)
        logging.info('Testing scaled_to_df: SUCCESS')
    except AssertionError as err:
        logging.warning(
            'Testing scaled_to_df: Scaled X_train is of incorrect type')
        raise err
    return X_scaled_to_df


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)

    path = './models/'
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.warning('Testing train_models: Results image files not found')
        raise err

    path = './images/results'
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.warning('Testing train_models: Model files not found')
        raise err


if __name__ == "__main__":
    DATA_FRAME = test_import(clu.import_data)
    test_eda(clu.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(clu.encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        clu.perform_feature_engineering, DATA_FRAME)
    X_train_scaled = test_X_scaler(clu.X_scaler, X_TRAIN)
    test_scaled_to_df(clu.scaled_to_df, X_train_scaled)


    test_train_models(
        clu.train_models,
        X_TRAIN,
        X_TEST,
        Y_TRAIN,
        Y_TEST)
