import argparse
import numpy
import src.utils.commons
import src.models.base_model
import plotly.plotly
import plotly.graph_objs
import plotly.offline
import typing

import sklearn.linear_model


def parse_args() -> typing.Dict:
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path",
                        default="../../data/processed/Grupos_totales_continua_features.csv")
    parser.add_argument("-o", "--output_path", type=str, help="output path",
                        default="../../reports/figures/ridge_tuning.html")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, output_path: str):

    scoring = {'rmse': sklearn.metrics.make_scorer(src.models.base_model.rmse)}
    X, y = src.utils.commons.load_features(input_path)
    parameter_value = []
    score_value = []
    for alpha in numpy.arange(0.01, 1., 0.05):
        model = sklearn.linear_model.Ridge(alpha=alpha)
        scores = sklearn.model_selection.cross_validate(estimator=model, X=X, y=y, cv=10, scoring=scoring)
        score_value.append(scores['test_rmse'].mean())
        parameter_value.append(alpha)

    data = [plotly.graph_objs.Scatter(x=parameter_value, y=score_value)]

    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Ridge Regression Parameter tunning',
        xaxis=dict(
            title='Alpha'
        ),
        yaxis=dict(
            title='Root Mean Squared Error'
        ),
    )
    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=output_path)


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)
