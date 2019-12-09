import pandas
import scipy.io
import argparse
import plotly.plotly
import plotly.graph_objs

from typing import List, Dict


def reduce_to_hours(df: pandas.DataFrame) -> List:
    """
    Given an input DataFrame obtain the fog by hour frequencies
    :param pandas.DataFrame df: input DataFrame it should contain day, hour and VRV values
    :return List vector: list with fog hours in the DataFrame
    """

    # 0 is day, 1 is hours
    _df = df.copy().sort_values(by=[0, 1])

    # loc valueswith fog
    _df.loc[:, 'fog'] = (_df[23] < 2000).astype(int)

    # create a dummy timestamp
    _df.loc[:, 'time'] = _df[0]*24 + _df[1]
    return pandas.DataFrame(_df.loc[_df['fog'] == 1, 'time'].diff().ge(1.5).cumsum()).groupby('time').size().to_list()


def fog_hour_frequencies(mat) -> List:
    """
    Obtain the fog by our frequencies in the whole mat
    :param mat: mat file descriptor
    :return List vector: list with all fog hours in the DataSet
    """
    fogfreq = list()
    for idx, dataset in enumerate(range(mat['Grupos_totales_continua'].shape[1])):
        df = pandas.DataFrame(mat['Grupos_totales_continua'][0][dataset])
        vector = reduce_to_hours(df)
        fogfreq.extend(vector)
    return fogfreq


def read_mat(input_file: str) -> Dict:
    """
    Reads a mat file descriptor a returns a dict
    :param str input_file: file path
    :return Dict: dict representation of the mat file
    """
    return scipy.io.loadmat(input_file)


def parse_args():
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="input path",
                        default="../../data/raw/Grupos_totales_continua.mat")
    parser.add_argument("-o", "--output", type=str, help="output path",
                        default="../../reports/figures/fog_distribution.html")
    args = parser.parse_args()
    return args


def generate_histogram(data: List, output_file: str):
    """
    Creates a normalized histogram given input data
     * X axis: fog duration in hours
     * Y axis: fog normalized frequency
    :param List data: input data
    :param str output_file: place to save the visualization
    """

    hist_data = [plotly.graph_objs.Histogram(x=data, histnorm='probability')]
    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Fog distribution',
        xaxis=dict(
            title='Fog duration [Hours]'
        ),
        yaxis=dict(
            title='Norm Count'
        ),
    )
    fig = plotly.graph_objs.Figure(data=hist_data, layout=layout)
    plotly.plotly.plot(fig, filename=output_file)


def main(input_path: str, ouptut_path: str):
    """
    read input path, transforms data and generates and histogram
    """
    mat = read_mat(input_path)
    fog_frequecies = fog_hour_frequencies()
    generate_histogram(fog_frequecies, ouptut_path)


if __name__ == '__main__':
    args = parse_args()
    main(**args)




