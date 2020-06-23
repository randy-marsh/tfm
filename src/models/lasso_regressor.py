import sklearn.linear_model
import src.models.base_model
import configparser


class LassoRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        config = configparser.ConfigParser().read('lasso.cfg')
        return sklearn.linear_model.Lasso(alpha=config.getfloat('DEFAULT', 'alpha'))

    @property
    def estimator_name(self):
        return 'Lasso Regression'