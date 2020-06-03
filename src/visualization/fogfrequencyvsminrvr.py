"""
Draw a scatter plot with the duration of the fog event versus the minimum RVR value
"""
import pandas
import argparse
import numpy
import os
import plotly.plotly
import plotly.graph_objs
import plotly.offline

from typing import List, Dict

import src.utils.commons
import src.features.make_features


def extract_minimum_rvr(df: pandas.DataFrame) -> List:
    """
    Given an input DataFrame obtain the minimum rvr per fog event
    :param pandas.DataFrame df: input DataFrame it should contain day, hour and VRV values
    :return List vector: list with fog hours in the DataFrame
    """
    dataset_with_fog_events = src.utils.commons.identify_fog_events(df)
    return dataset_with_fog_events.groupby('groups').apply(src.features.make_features.extract_labels).values.tolist()

    # return pandas.DataFrame(dataset_with_fog_events).groupby('groups').size().to_list()


def extract_fog_duration(df: pandas.DataFrame) -> List:
    """
    Given an input DataFrame obtain the fog by hour frequencies
    :param pandas.DataFrame df: input DataFrame it should contain day, hour and VRV values
    :return List vector: list with fog hours in the DataFrame
    """

    # # 0 is day, 1 is hours
    # _df = df.copy().sort_values(by=[0, 1])
    #
    # # loc values with fog
    # _df.loc[:, 'fog'] = (_df[23] < 2000).astype(int)
    #
    # # create a dummy timestamp
    # _df.loc[:, 'time'] = _df[0]*24 + _df[1]
    # patata = _df.loc[_df['fog'] == 1, 'time'].diff().ge(1.5).cumsum()
    dataset_with_fog_events = src.utils.commons.identify_fog_events(df)

    return pandas.DataFrame(dataset_with_fog_events).groupby('groups').size().to_list()


def fog_hour_frequencies(mat) -> numpy.array:
    """
    Obtain the fog duration and the minimun RVR per fog event
    :param mat: mat file descriptor
    :return numpy.array vector: list with all fog hours in the DataSet
    """
    foghourfreq = list()
    fogrvrmin = list()
    for dataset in src.utils.commons.dataset_generator(mat):
        foghourfreq.extend(extract_fog_duration(dataset))
        fogrvrmin.extend(extract_minimum_rvr(dataset))
    return numpy.column_stack((foghourfreq, fogrvrmin))


def generate_scatter(data: numpy.array, output_file: str):
    """
    Creates a normalized histogram given input data
     * X axis: fog duration in hours
     * Y axis: fog  frequency
    :param List data: input data
    :param str output_file: place to save the visualization
    """

    hist_data = [plotly.graph_objs.Scatter(x=data[:, 0], y=data[:, 1],
                                           mode='markers',
                                           marker=dict(size=15),
                                           )]
    # generate layout
    layout = plotly.graph_objs.Layout(
        title='Fog duration vs. Minimum RVR value',
        xaxis=dict(
            title='Fog duration [Hours]'
        ),
        yaxis=dict(
            title='Minimum RVR value'
        ),
    )
    fig = plotly.graph_objs.Figure(data=hist_data, layout=layout)
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
                        default="../../reports/figures/fog_durationvsminrvr.html")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, output_path: str):
    """
    read input path, transforms data and generates and histogram
    """
    mat = src.utils.commons.read_mat(input_path)
    fog_frequecies = fog_hour_frequencies(mat)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    generate_scatter(fog_frequecies, output_path)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
