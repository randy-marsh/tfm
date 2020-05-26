import sklearn.linear_model
import sklearn.model_selection
import numpy
import typing
import sklearn.metrics
import abc


def rmse(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> float:
    """
    Root mean squared error
    :param numpy.ndarray y_true: Ground truth (correct) target values.
    :param numpy.ndarray y_pred: Estimated target values.
    :return float: root mean squared error
    """
    return numpy.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred))

class BaseModel(abc.ABC):

    def __init__(self, X: numpy.ndarray, y: numpy.ndarray, cv: int) -> None:
        if X.shape[1] == 1:
            self._X = X.reshape(-1, 1)
        else:
            self._X = X
        self._y = y
        self._cv = cv
        self._scoring = {'root mean squared error': sklearn.metrics.make_scorer(rmse),
                         'mean absolute error': 'neg_mean_absolute_error',
                         'mean squared error': 'neg_mean_squared_error',
                         'coefficient of determination': 'r2',
                         }
        pass

    @property
    @abc.abstractmethod
    def estimator(self):
        pass

    def scores(self):
        sklearn.model_selection.cross_validate(estimator=self.estimator, X=self.X, y=self.y, cv=self.cv,
                                               scoring=self.scoring)
        pass

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


