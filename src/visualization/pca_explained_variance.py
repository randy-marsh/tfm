import pandas
import sklearn.decomposition
import numpy
import argparse
import plotly.plotly
import plotly.graph_objs
import plotly.offline
import os

from typing import Dict

import src.utils.commons

def generate_pca_explained_var(df: pandas.DataFrame, n_components: int, output_file: str):
    """
    Generates a  cumulative explained variance plot
    :param pandas.DataFrame df: input DataFrame
    :param int n_components: number of principal components to be performed
    :param str output_file: place to save the visualization
    """
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(df)
    pca_trace = [plotly.graph_objs.Scatter(x=list(range(1, n_components + 1)),
                                           y=numpy.cumsum(pca.explained_variance_ratio_))]
    pca_layout = plotly.graph_objs.Layout(
        title='PCA Analysis',
        xaxis=dict(
            title='number of components'
        ),
        yaxis=dict(
            title='Cumulative explained variance'
        ),
    )
    fig = plotly.graph_objs.Figure(data=pca_trace, layout=pca_layout)
    plotly.offline.plot(fig, filename=output_file)


def parse_args() -> Dict:
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path",
                        default="../../data/raw/Grupos_totales_continua.mat")
    parser.add_argument("-o", "--output_path", type=str, help="output path",
                        default="../../reports/figures/pca_cumulative_explained_variance.html")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, output_path: str):
    """
    read input path, transforms data and generates the pca visualization
    """
    mat = src.utils.commons.read_mat(input_path)
    dataset = src.utils.commons.concat_dataset(mat)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    generate_pca_explained_var(dataset.loc[:, [column for column in dataset.columns
                                               if column not in [0, 1, 23]]], n_components=10, output_file=output_path)


if __name__ == '__main__':
    args = parse_args()
    main(**args)