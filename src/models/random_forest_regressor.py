import sklearn.ensemble
import src.models.base_model


class RandomForestRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        # TODO add hyperparameters from config file
        return sklearn.ensemble.RandomForestRegressor()

    @property
    def estimator_name(self):
        return 'Random Forest Regression'