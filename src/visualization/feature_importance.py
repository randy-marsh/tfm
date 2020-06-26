"""
Feature importance based on random forest.
Train a model an plot its importance
"""

import argparse
import typing
import numpy
import pandas
import plotly.plotly
import plotly.graph_objs
import plotly.offline

import sklearn.ensemble

import src.utils.commons


def parse_args() -> typing.Dict:
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path",
                        default="../../data/processed/Grupos_totales_continua_features.csv")
    parser.add_argument("-f", "--feature_path", type=str, help="path to the excel with the feature names",
                        default="../../data/raw/Datos_CIBA_Torre.xlsx")
    parser.add_argument("-o", "--output_path", type=str, help="output path",
                        default="../../reports/figures/feature_importance.html")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, feature_path: str, output_path: str):
    # feature names
    names = pandas.read_excel(feature_path, nrows=0).columns.tolist()
    names = names[2:]
    names.append('first RVR')

    X, y = src.utils.commons.load_features(input_path)

    # Build a forest and compute the impurity-based feature importances
    forest = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=None,  random_state=0)

    forest.fit(X, y)

    importances = forest.feature_importances_
    std = numpy.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = numpy.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print(f"{f + 1} feature {names[indices[f]]} ({importances[indices[f]]})")

    data = [plotly.graph_objs.Bar(x=[names[i] for i in indices], y=importances[indices])]

    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Random Forest based Feature Importance',
        xaxis=dict(
            title='Feature name'
        ),
        yaxis=dict(
            title='Gini importance'
        ),
    )
    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=output_path)


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
