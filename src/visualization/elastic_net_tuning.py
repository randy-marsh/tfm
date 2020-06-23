import argparse
import numpy
import src.utils.commons
import src.models.base_model
import plotly.plotly
import plotly.graph_objs
import plotly.offline
import typing
import os

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
                        default="../../reports/figures/elastic_net_tuning.html")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, output_path: str):

    scoring = {'rmse': sklearn.metrics.make_scorer(src.models.base_model.rmse)}
    X, y = src.utils.commons.load_features(input_path)

    score_matrix = numpy.zeros((20, 20))
    # alpha idx i, l1_ratio_idx j
    for alpha_idx, alpha in enumerate(numpy.arange(0.01, 1., 0.05)):
        for l1_ratio_idx, l1_ratio in enumerate(numpy.arange(0.01, 1., 0.05)):
            model = sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            scores = sklearn.model_selection.cross_validate(estimator=model, X=X, y=y, cv=10, scoring=scoring)
            score_matrix[alpha_idx][l1_ratio_idx] = scores['test_rmse'].mean()


    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    index = numpy.where(score_matrix == score_matrix.min())
    print(f"index: {numpy.where(score_matrix == score_matrix.min())}")
    print(f"best value: {score_matrix.min()}")
    # print(f" alpha: {parameter_alpha_value[index]}, l1_ratio: {parameter_l1_ratio_value[index]}")
    print(f"alpha: {numpy.arange(0.01, 1., 0.05)[index[0]]}")
    print(f"l1: {numpy.arange(0.01, 1., 0.05)[index[1]]}")


    data = [plotly.graph_objs.Contour(z=score_matrix,
                                      x=numpy.arange(0.01, 1., 0.05),
                                      y=numpy.arange(0.01, 1., 0.05)
                                     )]

    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Elastic Net Regression Parameter tuning',
        xaxis=dict(
            title='l1_ratio'
        ),
        yaxis=dict(
            title='Alpha'
        ),
    )
    fig = plotly.graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=output_path, )


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)