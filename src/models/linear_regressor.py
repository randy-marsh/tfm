import sklearn.linear_model
import src.models.base_model


class LinearRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        return sklearn.linear_model.LinearRegression()

    @property
    def estimator_name(self):
        return 'Linear Regression'