import pandas
import scipy.io

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




