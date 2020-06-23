import sklearn.linear_model
import src.models.base_model
import configparser


class LassoRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        try:
            config = configparser.RawConfigParser()
            config.read('lasso.cfg')
            return sklearn.linear_model.Lasso(alpha=config.getfloat('DEFAULT', 'alpha'))
        
        except configparser.NoOptionError or FileNotFoundError:
            # TODO warning or logging
            return sklearn.linear_model.Ridge()

    @property
    def estimator_name(self):
        return 'Lasso Regression'