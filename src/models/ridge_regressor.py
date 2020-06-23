import sklearn.linear_model
import src.models.base_model
import configparser


class RidgeRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        config = configparser.ConfigParser().read('ridge.cfg')
        return sklearn.linear_model.Ridge(alpha=config.getfloat('DEFAULT', 'alpha'))

    @property
    def estimator_name(self):
        return 'Ridge Regression'