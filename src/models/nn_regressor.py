import sklearn.neural_network
import src.models.base_model


class NNRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        # TODO add config
        return sklearn.neural_network.MLPRegressor()

    @property
    def estimator_name(self):
        return 'Neural Network Regression'