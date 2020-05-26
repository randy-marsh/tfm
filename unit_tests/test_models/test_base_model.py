import unittest
import  numpy
import src.models.base_model

def remove_abstract_methods_from_class(cls: 'A class'):
    cls_without_abstract_methods = cls
    cls_without_abstract_methods.__abstractmethods__ = set()
    return cls_without_abstract_methods
class TestRmse(unittest.TestCase):

    def test_that_rmse_computes_root_mean_squared_error(self):
        y_pred = numpy.array([3, -0.5, 2, 7])
        y_true = numpy.array([2.5, 0.0, 2, 8])
        expected_value = 0.6123724356957945
        output_value = src.models.base_model.rmse(y_pred=y_pred, y_true=y_true)
        self.assertAlmostEqual(expected_value, output_value)


class TestBaseModelProperties(unittest.TestCase):

    def setUp(self) -> None:
        self.base_model_no_abstract = remove_abstract_methods_from_class(src.models.base_model.BaseModel)
        self.base_model = self.base_model_no_abstract(X=numpy.array([[1, 2, 3], [4, 5, 6]]), y=numpy.array([0, 1, 0]),
                                                      cv=10)

    def test_that_X_can_be_read(self):
        self.assertEqual(numpy.array_equal(self.base_model.X, numpy.array([[1, 2, 3], [4, 5, 6]])), True)

    def test_that_y_can_be_read(self):
        self.assertEqual(numpy.array_equal(self.base_model.y,  numpy.array([0, 1, 0])), True)

    def test_that_cv_can_be_read(self):
        self.assertEqual(self.base_model.cv, 10)