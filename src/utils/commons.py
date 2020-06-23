import pandas
import scipy.io
import typing
import sklearn.preprocessing
from typing import Dict


def read_mat(input_file: str) -> Dict:
    """
    Reads a mat file descriptor a returns a dict
    :param str input_file: file path
    :return Dict: dict representation of the mat file
    """
    return scipy.io.loadmat(input_file)


def concat_dataset(mat: Dict) -> pandas.DataFrame:
    """
    Concat all datasets into one
    :param mat: mat object descriptor
    :return pandas.DataFrame: DataFrame containing all datasets
    """
    # TODO reformat
    datasets = list()
    for idx, dataset in enumerate(range(mat['Grupos_totales_continua'].shape[1])):
        df = pandas.DataFrame(mat['Grupos_totales_continua'][0][dataset])
        datasets.append(df)
    return pandas.concat(datasets, ignore_index=True)


def dataset_generator(mat):
    """
    Common generator for iterate over datasets
    """
    data_key = get_mat_data_key(mat)
    if data_key == 'Grupos_totales_continua':
        for dataset in range(mat[data_key].shape[1]):
            df = pandas.DataFrame(mat[data_key][0][dataset])
            df = standarize_dataset(df, data_key)
            yield df
    else:
        df = pandas.DataFrame(mat[data_key])
        df = standarize_dataset(df, data_key)
        yield df


def get_mat_data_key(mat: Dict) -> str:
    """
    Obtain data key in a mat dict
    :param Dict mat: mat dict descriptor
    :return str: str with the data key
    """
    mat_keys = list(mat.keys())
    data_key = [valid_keys for valid_keys in mat_keys if valid_keys not in ['__header__', '__version__', '__globals__']]
    return data_key[0]


def standarize_dataset(df: pandas.DataFrame, origin: str) -> pandas.DataFrame:
    """
    Standarize the dataset given its origin
    :param pandas.DataFrame df: raw dataframe
    :param str origin: dataset origin valid values 'Grupos_totales_continua', 'Visib_A8_Todas_h'
    :return pandas.DataFrame: a new dataframe with renamed columns 'rvr' as runway visual range
    :raises: NotImplemented Error if origin does not match
    """
    if origin == 'Grupos_totales_continua':

        # drop 0 and 1 columms
        df.drop(columns=[0, 1], inplace=True)

        # rename 23 to rvr
        df.rename(columns={23: 'rvr'}, inplace=True)
    elif origin == 'Visib_A8_Todas_h':
        # rename 2 to rvr
        df.rename(columns={2: 'rvr'}, inplace=True)
    else:
        raise NotImplementedError(f"{origin} is not in 'Grupos_totales_continua', 'Visib_A8_Todas_h'")
    return df


def identify_fog_events(df: pandas.DataFrame) -> pandas.DataFrame:
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

def load_features(input_data: str ) -> typing.Tuple:
    """
    Load data and normalize it
    :param str input_data: input data path

    :return: tuple containing (X, y)
    """

    dataset = pandas.read_csv(input_data, sep=';')

    X = dataset.loc[:, [column for column in dataset.columns if column not in ['id', 'y']]]
    y = dataset['y'].values

    # Normalize features
    X = sklearn.preprocessing.scale(X)
    return X, y





