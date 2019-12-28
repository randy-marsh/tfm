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


def concat_dataset(mat) -> pandas.DataFrame:
    """
    Concat all datasets into one
    :param mat: mat object descriptor
    :return pandas.DataFrame: DataFrame containing all datasets
    """
    datasets = list()
    for idx, dataset in enumerate(range(mat['Grupos_totales_continua'].shape[1])):
        df = pandas.DataFrame(mat['Grupos_totales_continua'][0][dataset])
        datasets.append(df)
    return pandas.concat(datasets, ignore_index=True)

def dataset_generator(mat):
    """
    Common generator for iterate over datasets
    """
    for dataset in range(mat['Grupos_totales_continua'].shape[1]):
        df = pandas.DataFrame(mat['Grupos_totales_continua'][0][dataset])
        yield df

