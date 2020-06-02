import sklearn.linear_model
import src.models.base_model


class LassoRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        # TODO add alpha from config file
        return sklearn.linear_model.Lasso()

    @property
    def estimator_name(self):
        return 'Lasso Regression'