import argparse
import numpy
import src.utils.commons
import src.models.base_model
import plotly.plotly
import plotly.graph_objs
import plotly.offline
import typing
import os

import sklearn.ensemble


def parse_args() -> typing.Dict:
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path",
                        default="../../data/processed/Grupos_totales_continua_features.csv")
    parser.add_argument("-o", "--output_path", type=str, help="output path",
                        default="../../reports/figures/random_forest_tuning.html")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, output_path: str):

    scoring = {'rmse': sklearn.metrics.make_scorer(src.models.base_model.rmse)}
    X, y = src.utils.commons.load_features(input_path)

    x_range = [i for i in range(1, 21, 1)]
    y_range = [i for i in range(1, 11, 1)]
    score_matrix = numpy.zeros((len(y_range), len(x_range)))
    # estimators_idx i, depth_idx j
    for estimators_idx, estimator in enumerate(y_range):
        for depth_idx, depth in enumerate(x_range):
            model = sklearn.ensemble.RandomForestRegressor(n_estimators=estimator, max_depth=depth)
            scores = sklearn.model_selection.cross_validate(estimator=model, X=X, y=y, cv=10, scoring=scoring)
            score_matrix[estimators_idx][depth_idx] = scores['test_rmse'].mean()

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    index = numpy.where(score_matrix == score_matrix.min())
    print(f"index: {numpy.where(score_matrix == score_matrix.min())}")
    print(f"best value: {score_matrix.min()}")
    print(f"Estimator: {y_range[index[0]]}")
    print(f"max_depth: {x_range[index[1]]}")

    data = [plotly.graph_objs.Contour(z=score_matrix,
                                      x=x_range,
                                      y=y_range
                                      )]

    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Random forest Parameter tuning',
        xaxis=dict(
            title='Max depth'
        ),
        yaxis=dict(
            title='Estimators'
        ),
    )
    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=output_path, )


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)