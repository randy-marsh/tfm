import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import numpy
import typing
import sklearn.metrics
import abc
import os.path
import pandas


def rmse(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
    """
    Root mean squared error
    :param numpy.ndarray y_true: Ground truth (correct) target values.
    :param numpy.ndarray y_pred: Estimated target values.
    :return float: root mean squared error
    """
    return numpy.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))


def threshold_accuracy(y_true: numpy.ndarray, y_pred: numpy.ndarray, thr: float or int) -> float:
    """
    Umbralizes the input given a threshold and compute the accuracy
    :param numpy.ndarray y_true: Ground truth (correct) target values.
    :param numpy.ndarray y_pred: Estimated target values.
    :param thr: threshold
    :return float: accuracy estimation
    """
    y_true_thr = numpy.where(y_true <= thr, 1, 0)
    y_pred_thr = numpy.where(y_pred <= thr, 1, 0)
    return sklearn.metrics.accuracy_score(y_true=y_true_thr, y_pred=y_pred_thr)

class BaseModel(abc.ABC):

    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, cv: int = 10) -> None:

        self._X = X
        if X.ndim == 1:
            self._X = X.reshape(-1, 1)
        self._y = y
        self._cv = cv
        self._scoring = {'root mean squared error': sklearn.metrics.make_scorer(rmse),
                         # 'mean absolute error': 'neg_mean_absolute_error',
                         'mean absolute error': sklearn.metrics.make_scorer(sklearn.metrics.mean_absolute_error),
                         # 'mean squared error': 'neg_mean_squared_error',
                         'mean squared error': sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error),
                         'coefficient of determination': 'r2',
                         }

    @property
    @abc.abstractmethod
    def estimator(self):
        pass

    @property
    @abc.abstractmethod
    def estimator_name(self):
        pass

    def make_scores(self):
        scores = sklearn.model_selection.cross_validate(estimator=self.estimator, X=self.X, y=self.y, cv=self.cv,
                                                        scoring=self.scoring, n_jobs=-1)
        self._scores = scores

    def to_csv(self, path: str, sep: str = ';'):
        data_dict = {'Model': [self.estimator_name],
                     'root mean squared error': [self.scores['test_root mean squared error'].mean()],
                     'mean absolute error': [self.scores['test_mean absolute error'].mean()],
                     'mean squared error': [self.scores['test_mean squared error'].mean()],
                     'coefficient of determination': [self.scores['test_coefficient of determination'].mean()],
                     }
        df = pandas.DataFrame(data=data_dict)
        if os.path.isfile(path):
            df.to_csv(path_or_buf=path, sep=sep, mode='a', header=False, index=False)
        else:
            df.to_csv(path_or_buf=path, sep=sep, index=False)

    def __str__(self):
        return (f"""{self.estimator_name}, root mean squared error: {self.scores['test_root mean squared error'].mean()}, """
                f"""mean absolute error: {self.scores['test_mean absolute error'].mean()}, """
                f"""mean squared error: {self.scores['test_mean squared error'].mean()}, """
                f"""coefficient of determination: {self.scores['test_coefficient of determination'].mean()}""")

    @property
    def X(self) -> numpy.ndarray:
        return self._X

    @property
    def y(self) -> numpy.ndarray:
        return self._y

    @property
    def cv(self) -> int:
        return self._cv

    @property
    def scoring(self) -> typing.Dict:
        return self._scoring

    @property
    def scores(self) -> typing.Dict:
        try:
            return self._scores
        except AttributeError:
            self.make_scores()
            return self._scores

