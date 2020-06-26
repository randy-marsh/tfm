import sklearn.linear_model
import src.models.base_model
import configparser


class ElasticNetRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        try:
            config = configparser.RawConfigParser()
            config.read('elastic_net.cfg')
            alpha = config.getfloat('DEFAULT', 'alpha')
            l1_ratio = config.getfloat('DEFAULT', 'l1_ratio')
            return sklearn.linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        except configparser.NoOptionError:
            # TODO warning or logging
            return sklearn.linear_model.ElasticNet()

        except FileNotFoundError:
            # TODO warning or logging
            return sklearn.linear_model.ElasticNet()

        except configparser.MissingSectionHeaderError:
            # TODO warning or logging
            return sklearn.linear_model.ElasticNet()

    @property
    def estimator_name(self):
        return 'Elastic Net Regression'