import unittest
import numpy
import pandas
import tempfile
import sklearn.linear_model

import src.models.base_model

def remove_abstract_methods_from_class(cls: 'A class'):
    cls_without_abstract_methods = cls
    cls_without_abstract_methods.__abstractmethods__ = set()
    return cls_without_abstract_methods


class DummyRegressor(src.models.base_model.BaseModel):

    @property
    def estimator(self):
        return sklearn.linear_model.LinearRegression()

    @property
    def estimator_name(self):
        return 'Dummyr Regression'

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

    def test_that_scores_can_be_read(self):
        self.assertTrue(self.base_model.scores)


class TestToCsv(unittest.TestCase):

    def setUp(self) -> None:
        self.path = tempfile.TemporaryDirectory()

        self.model = DummyRegressor(X=numpy.array([numpy.random.random() for i in range(100)]),
                                    y=numpy.array([numpy.random.random() for i in range(100)]),
                                    cv=10)

    def tearDown(self) -> None:
        self.path.cleanup()

    def test_that_output_csv_has_correct_column_names(self):
        self.model.to_csv(path=self.path.name + '\dummy.csv')
        df = pandas.read_csv(self.path.name + '\dummy.csv', sep=';')
        self.assertListEqual(sorted(df.columns.tolist()), sorted(['Model', 'root mean squared error',
                                                                  'mean absolute error', 'mean squared error',
                                                                  'coefficient of determination']))

    def test_that_if_file_exists_then_appends_only_one_row(self):
        self.model.to_csv(path=self.path.name + '\dummy.csv')
        df = pandas.read_csv(self.path.name + '\dummy.csv', sep=';')
        rows_before = df.shape[0]
        self.model.to_csv(path=self.path.name + '\dummy.csv')
        df = pandas.read_csv(self.path.name + '\dummy.csv', sep=';')
        rows_after = df.shape[0]
        self.assertEqual(rows_before + 1, rows_after)

    def test_that_if_file_does_not_exists_then_adds_one_row(self):
        self.model.to_csv(path=self.path.name + '\dummy.csv')
        df = pandas.read_csv(self.path.name + '\dummy.csv', sep=';')
        rows = df.shape[0]
        self.assertEqual(rows, 1)


class TestThresholdAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self.thr = 1000

    def test_that_if_all_values_are_greater_than_1000_then_output_is_one(self):
        y_pred = numpy.array([2000, 2000, 2000, 2000])
        y_true = numpy.array([2000, 2000, 2000, 2000])
        self.assertEqual(1, src.models.base_model.threshold_accuracy(y_true=y_true, y_pred=y_pred, thr=self.thr))

    def test_that_if_y_pred_are_less_than_1000_and_y_true_is_greater_than_1000_then_output_is_zero(self):
        y_pred = numpy.array([1, 1, 1, 1])
        y_true = numpy.array([2000, 2000, 2000, 2000])
        self.assertEqual(0, src.models.base_model.threshold_accuracy(y_true=y_true, y_pred=y_pred, thr=self.thr))

    def test_that_if_y_true_are_less_than_1000_and_y_pred_is_greater_than_1000_then_output_is_zero(self):
        y_true = numpy.array([1, 1, 1, 1])
        y_pred = numpy.array([2000, 2000, 2000, 2000])
        self.assertEqual(0, src.models.base_model.threshold_accuracy(y_true=y_true, y_pred=y_pred, thr=self.thr))

    def test_that_if_all_values_are_equal_than_1000_then_output_is_one(self):
        y_pred = numpy.array([1000, 1000, 1000, 1000])
        y_true = numpy.array([1000, 1000, 1000, 1000])
        self.assertEqual(1, src.models.base_model.threshold_accuracy(y_true=y_true, y_pred=y_pred, thr=self.thr))

    def test_that_if_y_true_is_allways_1000_and_y_pred_is_greater_to_1000_50_percent_of_time_then_output_is_0_5(self):
        y_pred = numpy.array([2000, 1, 2000, 1])
        y_true = numpy.array([1000, 1000, 1000, 1000])
        self.assertEqual(0.5, src.models.base_model.threshold_accuracy(y_true=y_true, y_pred=y_pred, thr=self.thr))




