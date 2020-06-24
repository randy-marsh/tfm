import sklearn.ensemble
import src.models.base_model
import configparser


class RandomForestRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        try:
            config = configparser.RawConfigParser()
            config.read('random_forest.cfg')
            n_estimators = config.getint('DEFAULT', 'n_estimators')
            max_depth = config.getint('DEFAULT', 'max_depth')
            return sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        except configparser.NoOptionError or FileNotFoundError:
            # TODO warning or logging
            return sklearn.ensemble.RandomForestRegressor()


    @property
    def estimator_name(self):
        return 'Random Forest Regression'