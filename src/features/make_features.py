import pandas
import numpy
import scipy.io
import os
import argparse
import sklearn.decomposition
import sklearn.model_selection
import tsfresh
import tsfresh.feature_extraction.settings
from typing import Iterator, Generator, Dict

import src.utils.commons


def identify_log_events(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Given an input DataFrame identify the fog by hour frequencies
    :param pandas.DataFrame df: input DataFrame it should contain day, hour and RVR values
    :return pandas.DataFrame df: DataFrame with an extra column 'groups' with the fog-events labeled
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


def extract_sequences(df: pandas.DataFrame, window_lenght: int, fog_event_lenght:int,
                      id_generator: Generator, pca_model) -> pandas.DataFrame:
    """
    Extract sequences of window_lenght from fog events greater than fog_event_lenght
    """
    # get number of events
    unique_fog_events_labels = df['groups'].unique()
    fog_events = unique_fog_events_labels[~numpy.isnan(unique_fog_events_labels)]
    acumulated_sequences = []
    # TODO reformat this function
    valid_sequences = 0
    for fog_event in fog_events:
        if len(df.loc[df['groups'] == fog_event]) >= fog_event_lenght:
            valid_sequences += 1
            # get indexes
            _index = df.loc[df['groups'] == fog_event].index
            # transform the dataset
            _df = pandas.DataFrame(pca_model.transform(df.copy().loc[:, [column for column in df.columns
                                                                         if column not in [0, 1, 23, 'fog',
                                                                                         'time', 'groups']]]))
            # get window_lenght before the fog event
            sequence = _df.loc[pandas.RangeIndex(_index[0] - (window_lenght - 1), _index[-1])]

            # add a time index
            sequence.loc[:, 'time'] = sequence.reset_index().index
            # TODO this is not he optimal way to do it
            sequence.loc[:, 'id'] = next(id_generator)
            acumulated_sequences.append(sequence)
    if valid_sequences is 0:
        return None
    else:
        return pandas.concat(acumulated_sequences, ignore_index=True)


def extract_labels(df: pandas.DataFrame, fog_event_lenght) -> pandas.DataFrame:
    # get number of events
    unique_fog_events_labels = df['groups'].unique()
    fog_events = unique_fog_events_labels[~numpy.isnan(unique_fog_events_labels)]
    acumulated_labels = list()
    # TODO reformat this function b
    for fog_event in fog_events:
        group_df = df.loc[df['groups'] == fog_event]
        if len(group_df) >= fog_event_lenght:

            # y is the minimum RVR in the fog event
            y = group_df[23].min()
            acumulated_labels.append(y)
    return acumulated_labels



def id_generator() -> Iterator[int]:
    """
    Dummy generator to keep track of the sequences
    """
    id = 0
    while True:
        yield id
        id += 1


def extract_characteristics_from_sequences(df: pandas.DataFrame, settings) -> pandas.DataFrame:
    """

    """
    # TODO reduce extracted features, see impute or features dict
    return tsfresh.extract_features(df, column_id='id', column_sort='time', default_fc_parameters=settings)


def parse_args() -> Dict:
    """
    Default arg parser
    reads input and output path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="input path",
                        default="../../data/raw/Grupos_totales_continua.mat")
    parser.add_argument("-o", "--output_path", type=str, help="output path",
                        default="../../data/processed/")
    args = parser.parse_args()
    return vars(args)


def main(input_path: str, output_path: str):
    """
    read input path, transforms data and generates features
    """
    # generator used to iterate over the dataset
    generator = id_generator()

    all_sequences = list()
    labels = list()
    settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
    mat = src.utils.commons.read_mat(input_path)
    datasets = src.utils.commons.concat_dataset(mat)
    # TODO generate a representative sample
    pca = sklearn.decomposition.PCA(n_components=4)
    pca.fit(datasets.loc[:, [column for column in datasets.columns
                             if column not in [0, 1, 23]]])
    for dataset in src.utils.commons.dataset_generator(mat):
        dataset_with_fog_events = identify_log_events(dataset)
        labels.extend(extract_labels(dataset_with_fog_events, fog_event_lenght=2))
        dataset_sequences = extract_sequences(dataset_with_fog_events, window_lenght=10, fog_event_lenght=2,
                                              id_generator=generator, pca_model=pca)
        all_sequences.append(dataset_sequences)


    # generate a single DataFrame with all the sequences
    datasets_sequences = pandas.concat(all_sequences, ignore_index=False)

    datasets_features = extract_characteristics_from_sequences(datasets_sequences.dropna(), settings=settings)

    # add labels to dataset
    datasets_features.loc[:, 'y'] = labels
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    datasets_features.to_csv(output_path + 'features.csv', sep=';', index=True)

    # TODO generate labels


if __name__ == '__main__':
    args = parse_args()
    main(**args)













