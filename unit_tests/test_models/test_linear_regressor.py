import unittest
import numpy
import src.models.linear_regressor

class TestLinearRegressorProperties(unittest.TestCase):

    def setUp(self) -> None:
        self.model = src.models.linear_regressor.LinearRegressor(X=numpy.array([[1, 2, 3], [4, 5, 6]]),
                                                                 y=numpy.array([0, 1, 0]), cv=10)

    def test_that_X_can_be_read(self):
        self.assertEqual(numpy.array_equal(self.model.X, numpy.array([[1, 2, 3], [4, 5, 6]])), True)

    def test_that_y_can_be_read(self):
        self.assertEqual(numpy.array_equal(self.model.y,  numpy.array([0, 1, 0])), True)

    def test_that_cv_can_be_read(self):
        self.assertEqual(self.model.cv, 10 )

    def test_that_estimator_name_can_be_read(self):
        self.assertEqual(self.model.estimator_name, 'Linear Regression')