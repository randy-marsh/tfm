import sklearn.neural_network
import src.models.base_model
import configparser

class NNRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        try:
            #TODO there is a bug
            config = configparser.RawConfigParser()
            config.read('nn.cfg')
            alpha = config.getfloat('DEFAULT', 'alpha')
            hidden_layer_sizes = config.getint('DEFAULT', 'hidden_layer_sizes')
            solver = config.get('DEFAULT', 'solver')
            early_stopping = config.getboolean('DEFAULT', 'early_stopping')
            return sklearn.neural_network.MLPRegressor(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, max_iter=500000,
                                                       solver=solver, early_stopping=early_stopping)
        except configparser.NoOptionError or FileNotFoundError:
            # TODO warning or logging
            return sklearn.neural_network.MLPRegressor()

    @property
    def estimator_name(self):
        return 'Neural Network Regression'