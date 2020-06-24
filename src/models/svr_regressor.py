import sklearn.svm
import src.models.base_model
import configparser


class SVRRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        try:
            config = configparser.RawConfigParser()
            config.read('knn.cfg')
            kernel = config.get('DEFAULT', 'kernel')
            C = config.getfloat('DEFAULT', 'C')
            epsilon = config.getfloat('DEFAULT', 'epsilon')
            return sklearn.svm.SVR(kernel=kernel, C=C, epsilon=epsilon)
        except configparser.NoOptionError or FileNotFoundError:
            # TODO warning or logging
            return sklearn.svm.SVR()
        
    @property
    def estimator_name(self):
        return 'SVR Regression'