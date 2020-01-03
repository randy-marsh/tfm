import pandas
import numpy
import os
import argparse
import sklearn.decomposition
import sklearn.model_selection
import tsfresh
import tsfresh.feature_extraction.settings
import tsfresh.utilities.dataframe_functions
from typing import Iterator, Generator, Dict, List

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

def is_vrv_min_at_first_fog_event(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    """
    min_rvr = df[23].min()
    first_rvr = df.iloc[0][23]
    if min_rvr == first_rvr:
        return False
    else:
        return True

def get_fog_events(df: pandas.DataFrame) -> List:
    """
    read all the fog events in the DataFrame an returns a list with those groups
    """
    unique_fog_events_labels = df['groups'].unique()
    fog_events = unique_fog_events_labels[~numpy.isnan(unique_fog_events_labels)]
    return fog_events.tolist()


def valid_fog_events(df: pandas.DataFrame, min_fog_event_length: int, max_fog_event_length: int or None,
                     first_vrv: bool):
    """
    Yields valid fog_events given params
    """
    # sanity check
    if max_fog_event_length is not None:
        if min_fog_event_length >= max_fog_event_length:
            raise ValueError("max_window_length should be greater than min_window_length")
    fog_events = get_fog_events(df)
    for fog_event in fog_events:
        fog_event_df = df.loc[df['groups'] == fog_event]
        fog_event_df = fog_event_df.loc[:, [column for column in fog_event_df.columns if column != 'groups']]
        if len(fog_event_df) >= min_fog_event_length:
            if max_fog_event_length is not None:
                if len(fog_event_df) < max_fog_event_length:
                    if first_vrv:
                        if is_vrv_min_at_first_fog_event(fog_event_df):
                            yield fog_event_df
                    else:
                        yield fog_event_df
            elif first_vrv:
                if is_vrv_min_at_first_fog_event(fog_event_df):
                    yield fog_event_df
            else:
                yield fog_event_df


def extract_sequences(df: pandas.DataFrame, fog_events_df: pandas.DataFrame, window_length: int,
                      id_generator: Generator) -> pandas.DataFrame:
    """
    Extract sequences of window_length from fog events greater than fog_event_lenght
    """
    # get indexes
    _index = df.index

    # get window_length before the fog event
    sequence = fog_events_df.loc[pandas.RangeIndex(_index[0] - window_length, _index[0])]

    # remove column 'groups'
    sequence = sequence.loc[:, [column for column in sequence.columns if column != 'groups']]
    # add a time index
    sequence.loc[:, 'time'] = sequence.reset_index().index
    # TODO this is not he optimal way to do it
    sequence.loc[:, 'id'] = next(id_generator)
    return sequence


def extract_labels(df: pandas.DataFrame) -> int or float:
    """
    Get minimum RVR from input DataFrame
    :param pandas.DataFrame df: input DataFrame should contain a column named 23
    :return int: minimum RVR value
    """
    # y is the minimum RVR in the fog event
    # TODO this should be a function?
    y = df[23].min()
    return y


def get_first_vrv(df: pandas.DataFrame) -> int:
    """
    Get first RVR from input DataFrame
    :param pandas.DataFrame df: input DataFrame should contain a column named 23
    :return int: first RVR value
    """
    first_rvr = df.iloc[0][23]
    return first_rvr


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
    extracted_features = tsfresh.extract_features(df, column_id='id', column_sort='time',
                                                   default_fc_parameters=settings)
    tsfresh.utilities.dataframe_functions.impute(extracted_features)
    return extracted_features


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
    vrv = list()
    # settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
    settings = tsfresh.feature_extraction.settings.ComprehensiveFCParameters()
    # settings = tsfresh.feature_extraction.settings.EfficientFCParameters()
    mat = src.utils.commons.read_mat(input_path)
    datasets = src.utils.commons.concat_dataset(mat)
    # TODO generate a representative sample
    pca = sklearn.decomposition.PCA(n_components=4)

    # fit pca model
    pca.fit(datasets.loc[:, [column for column in datasets.columns
                             if column not in [0, 1, 23]]])

    # loop over the datasets
    for dataset in src.utils.commons.dataset_generator(mat):
        dataset_with_fog_events = identify_log_events(dataset)

        # apply pca
        fog_events_reduced = pandas.DataFrame(pca.transform(dataset_with_fog_events.loc[:,
                                                            [column for column in dataset_with_fog_events.columns
                                                             if column not in [23, 'time', 'id', 'groups']]]),
                                              index=dataset_with_fog_events.index)
        # loop over the groups
        for fog_event in valid_fog_events(dataset_with_fog_events, min_fog_event_length=2, max_fog_event_length=33,
                                          first_vrv=False):

            # get sequences
            sequences = extract_sequences(fog_event, fog_events_df=fog_events_reduced,
                                          window_length=10, id_generator=generator)
            all_sequences.append(sequences)

            # get y labels
            labels.append(extract_labels(fog_event))

            # get first vrv
            vrv.append(get_first_vrv(fog_event))

    # generate a single DataFrame with all the sequences
    datasets_sequences = pandas.concat(all_sequences, ignore_index=False)

    # extract features
    datasets_features = extract_characteristics_from_sequences(datasets_sequences.dropna(), settings=settings)

    # add labels to dataset
    datasets_features.loc[:, 'y'] = labels

    # add vrv to dataset
    datasets_features.loc[:, 'vrv'] = vrv

    # remove the mean
    datasets_features.loc[:, 'vrv'] = datasets_features['vrv'] - datasets_features['vrv'].mean()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    datasets_features.to_csv(output_path + 'features.csv', sep=';', index=False)
    # datasets_sequences.to_csv(output_path + 'features.csv', sep=';', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(**args)













