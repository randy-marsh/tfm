import sklearn.linear_model
import src.models.base_model
import configparser


class ElasticNetRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        config = configparser.ConfigParser().read('elastic_net.cfg')
        alpha = config.getfloat('DEFAULT', 'alpha')
        l1_ratio = config.getfloat('DEFAULT', 'l1_ratio')
        return sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    @property
    def estimator_name(self):
        return 'Elastic Net Regression'