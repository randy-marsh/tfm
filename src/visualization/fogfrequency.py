import pandas
import scipy.io
from typing import List


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

