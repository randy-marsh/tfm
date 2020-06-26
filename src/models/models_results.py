import argparse
import typing
import pandas
import os

import sklearn.preprocessing

import src.models.linear_regressor
import src.models.ridge_regressor
import src.models.lasso_regressor
import src.models.elastic_net_regressor

import src.models.random_forest_regressor
import src.models.knn_regressor
import src.models.svr_regressor
import src.models.nn_regressor


def parse_args() -> typing.Dict:
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser(description="Model evaluation trains models and evaluates its performance")
    parser.add_argument("-i", "--input_data", type=str, help="input data path",
                        default="../../data/processed/Grupos_totales_continua_features.csv")
    parser.add_argument("-o", "--output_file", type=str, help="results evaluation output path",
                        default="../../data/processed/model_results.csv")
    args = parser.parse_args()
    return vars(args)


def main(input_data: str, output_file: str) -> None:
    """
    Load data, normalize it and evaluate the models
    :param str input_data: input data path
    :param str output_file: path to store the results
    :return: None
    """

    dataset = pandas.read_csv(input_data, sep=';')

    X = dataset.loc[:, [column for column in dataset.columns if column not in ['id', 'y']]]
    y = dataset['y'].values

    # Normalize features
    X = sklearn.preprocessing.scale(X)

    models = [src.models.linear_regressor.LinearRegressor(X=X, y=y),
              src.models.ridge_regressor.RidgeRegressor(X=X, y=y),
              src.models.lasso_regressor.LassoRegressor(X=X, y=y),
              src.models.elastic_net_regressor.ElasticNetRegressor(X=X, y=y),
              src.models.knn_regressor.KNNRegressor(X=X, y=y),
              src.models.svr_regressor.SVRRegressor(X=X, y=y),
              src.models.random_forest_regressor.RandomForestRegressor(X=X, y=y),
              src.models.nn_regressor.NNRegressor(X=X, y=y),
    ]

    if os.path.isfile(output_file):
        os.remove(output_file)

    for model in models:
        model.to_csv(path=output_file)
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pandas.read_csv(output_file, sep=';'))

if __name__ == '__main__':
    args = parse_args()
    main(**args)

