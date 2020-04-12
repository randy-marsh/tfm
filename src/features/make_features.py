import pandas
import numpy
import os
import argparse
import sklearn
import sklearn.decomposition
import sklearn.model_selection
import tsfresh
import tsfresh.feature_extraction.settings
import tsfresh.utilities.dataframe_functions
from typing import Iterator, Generator, Dict, List
import pathlib
import src.utils.commons


def identify_log_events(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Given an input DataFrame
    """
    # do this to avoid modifying the original dataframe
    df = df.copy(deep=True)

    # loc values with fog column rvridentify the fog by hour frequencies, it asumes that the input dataframe is
    #     sorted by time and has a column called 'rvr'
    #     :param pandas.DataFrame df: input DataFrame it rvr values
    #     :return pandas.DataFrame df: DataFrame with an extra column 'groups' with the fog-events labeled
    df.loc[:, 'fog'] = (df['rvr'] < 2000).astype(int)

    # create a dummy timestamp
    df.loc[:, 'time'] = df.index

    # identify consecutive fog events and generates a unique label
    df.loc[:, 'groups'] = df.loc[df['fog'] == 1, 'time'].diff().ge(1.5).cumsum()
    # TODO think about return just a series rather than a whole dataframe or part of it
    # return everything but time vars
    return df.loc[:, [column for column in df.columns if column not in ['fog', 'time']]]


def is_rvr_min_at_first_fog_event(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Check whether minimum rvr is the first one or not
    :return: True if first rvr is not the first one, False otherwise
    """
    # TODO check the input
    min_rvr = df['rvr'].min()
    first_rvr = df.iloc[0]['rvr']
    if min_rvr == first_rvr:
        return False
    else:
        return True


def get_fog_events(df: pandas.DataFrame) -> List:
    """
    read all the fog events in the DataFrame an returns a list with those groups
    :param pandas.DataFrame df: dataframe with identified fog events
    :return List: list containing unique labeled fog events
    """
    unique_fog_events_labels = df['groups'].unique()
    fog_events = unique_fog_events_labels[~numpy.isnan(unique_fog_events_labels)]
    return fog_events.tolist()


def valid_fog_events(df: pandas.DataFrame, min_fog_event_length: int, max_fog_event_length: int or None,
                     first_rvr: bool):
    """
    Yields valid fog_events in input DataFrame given params
    :param pandas.DataFrame df: dataframe with identified fog events
    :param min_fog_event_length: fog event minimum length to filter by
    :param max_fog_event_length: fog event maximum length to filter by
    :param first_rvr: filter out fog events that its minimum matches the first rvr value
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
                    if first_rvr:
                        if is_rvr_min_at_first_fog_event(fog_event_df):
                            yield fog_event_df
                    else:
                        yield fog_event_df
            elif first_rvr:
                if is_rvr_min_at_first_fog_event(fog_event_df):
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


def extract_labels(df: pandas.DataFrame) -> float:
    """
    Get minimum RVR from input DataFrame
    :param pandas.DataFrame df: input DataFrame should contain a column named 23
    :return float: minimum RVR value
    """
    # y is the minimum RVR in the fog event
    # TODO this should be a function?
    y = df['rvr'].min()
    return y


def get_first_rvr(df: pandas.DataFrame) -> int:
    """
    Get first RVR from input DataFrame
    :param pandas.DataFrame df: input DataFrame should contain a column named 23
    :return int: first RVR value
    """

    first_rvr = df.iloc[0]['rvr']
    return first_rvr


def get_exogen_values(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Extract exogen vars first values
    :param pandas.DataFrame df: input dataframe
    :return pandas.Dataframe: dataframe with one roe but rvr values
    """
    out = df.head(1)
    return out.loc[:, [column for column in out.columns if column != 'rvr']]


def id_generator() -> Iterator[int]:
    """
    Dummy generator to keep track of the sequences
    """
    id = 0
    while True:
        yield id
        id += 1


def linear_features(df: pandas.DataFrame, column_id: str, column_sort: str) -> pandas.DataFrame:
    """
    Gets linear regression coefficients
    :param pandas.DataFrame df: input dataframe
    :param str column_id: column to identify events
    :param str column_sort: column to sort events
    :return pandas.Dataframe: a dataframe with a column for each coefficient
    """
    # TODO R wen not enough samples
    groups = df.groupby(column_id)
    lr = sklearn.linear_model.LinearRegression()
    results = dict()
    for name, group in groups:
        columns = [column for column in group.columns if column not in [column_id, column_sort]]
        clean = group.dropna()
        y = clean[column_sort]
        coefs = []
        for column in columns:
            X = clean[column]
            lr.fit(X.values.reshape(-1, 1), y)
            coefs.append(lr.coef_[0])
        results[name] = coefs
    # rename columns
    columns_renamed = ['coef_' + str(column) for column in df.columns if column not in [column_id, column_sort]]
    return pandas.DataFrame.from_dict(results, orient='index', columns=columns_renamed)


def extract_characteristics_from_sequences(df: pandas.DataFrame, settings) -> pandas.DataFrame:
    """
    Extract time series charateristics from the input dataframe
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
                        default="../../data/raw/Grupos_totales_continua")
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
    rvr = list()
    exogen_values = list()
    settings = tsfresh.feature_extraction.settings.MinimalFCParameters()
    # settings = tsfresh.feature_extraction.settings.ComprehensiveFCParameters()
    # settings = tsfresh.feature_extraction.settings.EfficientFCParameters()
    mat = src.utils.commons.read_mat(input_path)
    data_id = src.utils.commons.get_mat_data_key(mat)
    # datasets = src.utils.commons.concat_dataset(mat)
    # TODO generate a representative sample
    # pca = sklearn.decomposition.PCA(n_components=4)

    # fit pca model
    # pca.fit(datasets.loc[:, [column for column in datasets.columns
    #                          if column not in [0, 1, 23]]])

    # loop over the datasets
    for dataset in src.utils.commons.dataset_generator(mat):
        dataset_with_fog_events = identify_log_events(dataset)

        # apply pca
        # fog_events_reduced = pandas.DataFrame(pca.transform(dataset_with_fog_events.loc[:,
        #                                                     [column for column in dataset_with_fog_events.columns
        #                                                      if column not in [23, 'time', 'id', 'groups']]]),
        #                                       index=dataset_with_fog_events.index)
        fog_events_reduced = dataset_with_fog_events

        # loop over the groups
        for fog_event in valid_fog_events(dataset_with_fog_events, min_fog_event_length=2, max_fog_event_length=33,
                                          first_rvr=True):

            # get sequences
            sequences = extract_sequences(fog_event, fog_events_df=fog_events_reduced,
                                          window_length=10, id_generator=generator)

            all_sequences.append(sequences)

            # get y labels
            labels.append(extract_labels(fog_event))

            # get first rvr
            rvr.append(get_first_rvr(fog_event))

            # get exogen values
            exogen_values.append(get_exogen_values(fog_event))

    # generate a single DataFrame with all the sequences
    datasets_sequences = pandas.concat(all_sequences, ignore_index=False)
    # extract features
    # datasets_features = extract_characteristics_from_sequences(datasets_sequences.dropna(), settings=settings)
    dataset_exogen = pandas.concat(exogen_values, ignore_index=False)
    dataset_exogen.to_csv(output_path + 'exogen.csv', sep=';', index=False)
    # add labels to dataset
    # datasets_features.loc[:, 'y'] = labels

    # add rvr to dataset
    # datasets_features.loc[:, 'rvr'] = rvr
    # join exogen and features
    # datasets_features = pandas.concat([datasets_features, dataset_exogen], axis=1)
    # print(datasets_features.shape, dataset_exogen.shape)
    # datasets_features = pandas.merge(datasets_features.reset_index(drop=True), dataset_exogen.reset_index(drop=True),
    #                                  left_index=True, right_index=True)
    # remove the mean
    # datasets_features.loc[:, 'rvr'] = datasets_features['rvr'] - datasets_features['rvr'].mean()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # datasets_features.to_csv(output_path + 'features.csv', sep=';', index=False)

    coef_df = linear_features(datasets_sequences.loc[:, [column for column in datasets_sequences.columns
                                                         if 'rvr' != column]], 'id', 'time')
    # coef_df.loc[:, 'y'] = labels
    # coef_df.loc[:, 'rvr'] = rvr
    # coef_df.loc[:, 'rvr'] = coef_df['rvr'] - coef_df['rvr'].mean()
    coef_df.to_csv(output_path + data_id + '_coef.csv', sep=';', index=False)
    # datasets_features.loc[:, '0_coef'] = coef_df[0]
    # datasets_features.loc[:, '1_coef'] = coef_df[1]
    # datasets_features.loc[:, '2_coef'] = coef_df[2]
    # datasets_features.loc[:, '3_coef'] = coef_df[3]

    # extract last value from sequences
    # last_values = datasets_sequences.groupby('id').tail(1).iloc[:, :-2].reset_index(drop=True)
    # print(datasets_sequences.columns)
    # print(last_values.shape)
    # print(datasets_features.shape)
    # datasets_features.loc[:, '0_23last'] = last_values.iloc[:, 0]
    # datasets_features.loc[:, '1_23last'] = last_values.iloc[:, 1]
    # datasets_features.loc[:, '2_23last'] = last_values.iloc[:, 2]
    # datasets_features.loc[:, '3_23last'] = last_values.iloc[:, 3]
    # datasets_features.to_csv(output_path + 'features.csv', sep=';', index=False)
    # print(datasets_features.shape, dataset_exogen.shape)
    # datasets_sequences.to_csv(output_path + 'features.csv', sep=';', index=False)
    dataset_exogen.loc[:, 'y'] = labels
    dataset_exogen.loc[:, 'rvr'] = rvr
    dataset_exogen.to_csv(output_path + data_id + '_features.csv', sep=';', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(**args)













