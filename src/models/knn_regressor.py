import sklearn.neighbors
import src.models.base_model


class KNNRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        # TODO add config
        return sklearn.neighbors.KNeighborsRegressor()

    @property
    def estimator_name(self):
        return 'KNN Regression'