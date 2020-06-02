import sklearn.svm
import src.models.base_model


class SVRRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        # TODO add config
        return sklearn.svm.SVR()

    @property
    def estimator_name(self):
        return 'SVR Regression'