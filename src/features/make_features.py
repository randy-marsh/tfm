import pandas
import numpy
import sklearn.decomposition
import tsfresh
from typing import Iterator, Generator


def identify_log_events(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Given an input DataFrame identify the fog by hour frequencies
    :param pandas.DataFrame df: input DataFrame it should contain day, hour and RVR values
    :return pandas.DataFrame df: DataFrame whith an extra column 'groups' with the fog-events labeled
    """
    # do this to avoid modifying the original dataframe
    df = df.copy(deep=True)
    # loc values with fog column 23 is RVR
    df.loc[:, 'fog'] = (df[23] < 2000).astype(int)

    # create a dummy timestamp
    df.loc[:, 'time'] = df[0] * 24 + df[1]

    # identify consecutive fog events and generates a unique label
    df.loc[:, 'groups'] = df.loc[df['fog'] == 1, 'time'].diff().ge(1.5).cumsum()
    # TODO think about return just a series rather than a whole dataframe or part of it
    # return everything but time vars
    return df.loc[:, [column for column in df.columns if column not in [0, 1, 'fog', 'time']]]

def pca_filter(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Generate a pca filter
    """
    pass

def extract_sequences(df: pandas.DataFrame, window_lenght: int, fog_event_lenght:int, id_generator: Generator) -> pandas.DataFrame:
    """
    Extract sequences of window_lenght from fog events greater than fog_event_lenght
    """
    # get number of events
    unique_fog_events_labels = df['groups'].unique()
    fog_events = unique_fog_events_labels[~numpy.isnan(unique_fog_events_labels)]
    acumulated_sequences = []
    for fog_event in fog_events:
        if len(df.loc[df['groups'] == fog_event]) >= fog_event_lenght:

            # get indexes
            _index = df.loc[df['groups'] == fog_event].index
            # get window_lenght before the fog event
            sequence = df.loc[pandas.RangeIndex(_index[0] - (window_lenght - 1), _index[-1])]

            # add a time index
            sequence.loc[:, 'time'] = sequence.reset_index().index
            # TODO i think this is not he optimal way to do it
            sequence.loc[:, 'id'] = next(id_generator)
            acumulated_sequences.append(sequence)
    return pandas.concat(acumulated_sequences, ignore_index=True)

def id_generator() -> Iterator[int]:
    """
    Dummy generator
    """
        id = 0
        while True:
            yield id
            id += 1

def extract_characteristics_from_sequences(df: pandas.DataFrame) -> pandas.DataFrame:
    """

    """
    # TODO reduce extracted features, see impute or features dict
    return tsfresh.extract_features(df, column_id='id', column_sort='time')



