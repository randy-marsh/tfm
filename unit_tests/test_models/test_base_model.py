import unittest
import  numpy
import src.models.base_model

class TestRmse(unittest.TestCase):

    def test_that_rmse_computes_root_mean_squared_error(self):
        y_pred = numpy.array([3, -0.5, 2, 7])
        y_true = numpy.array([2.5, 0.0, 2, 8])
        expected_value = 0.6123724356957945
        output_value = src.models.base_model.rmse(y_pred=y_pred, y_true=y_true)
        self.assertAlmostEqual(expected_value, output_value)
