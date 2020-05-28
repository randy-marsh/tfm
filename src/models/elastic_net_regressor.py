import sklearn.linear_model
import src.models.base_model


class ElasticNetRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        # TODO add alpha from config file
        return sklearn.linear_model.ElasticNet()
    @property
    def estimator_name(self):
        return 'Elastic Net Regression'