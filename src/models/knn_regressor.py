import sklearn.neighbors
import src.models.base_model
import configparser


class KNNRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        try:
            config = configparser.RawConfigParser()
            config.read('knn.cfg')
            return sklearn.neighbors.KNeighborsRegressor(n_neighbors=config.getint('DEFAULT', 'n_neighbors'))
        except configparser.NoOptionError or FileNotFoundError:
            # TODO warning or logging
            return sklearn.neighbors.KNeighborsRegressor()

    @property
    def estimator_name(self):
        return 'KNN Regression'