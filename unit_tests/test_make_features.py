import unittest
import pandas
import numpy
import random
import src.features.make_features


class TestIdentifyFogEvents(unittest.TestCase):
    def test_that_if_input_has_one_long_fog_event_then_output_has_one_group(self):
        input_df = pandas.DataFrame({0: [1 for value in range(10)],
                                     1: [value for value in range(10)],
                                     # generate 1 fog-events
                                     23: [1 for value in range(10)]})
        output_df = src.features.make_features.identify_log_events(input_df)
        groups = output_df['groups'].unique()
        groups_len = len(groups[~numpy.isnan(groups)])
        self.assertEqual(groups_len, 1)

    def test_that_if_input_has_two_fog_events_separated_by_one_hour_then_output_has_two_groups(self):
        input_df = pandas.DataFrame({0: [1 for value in range(10)],
                                     1: [value for value in range(10)],
                                     # generate 2 fog-events
                                     23: ([1900 for value in range(5)] + [2000 for value in range(1)]
                                          + [1900 for value in range(4)])})
        output_df = src.features.make_features.identify_log_events(input_df)
        groups = output_df['groups'].unique()
        groups_len = len(groups[~numpy.isnan(groups)])
        self.assertEqual(groups_len, 2)

    def test_that_if_input_df_has_two_fog_events_separated_by_more_than_one_hour_then_output_has_two_groups(self):
        input_df = pandas.DataFrame({0: [1 for value in range(10)],
                                     1: [value for value in range(10)],
                                     # generate 2 fog-events
                                     23: ([2000 for value in range(3)] + [1900 for value in range(2)]
                                          + [2000 for value in range(2)] + [1900 for value in range(3)])})
        output_df = src.features.make_features.identify_log_events(input_df)
        groups = output_df['groups'].unique()
        groups_len = len(groups[~numpy.isnan(groups)])
        self.assertEqual(groups_len, 2)

    def test_that_output_not_contain_fog_or_time_columns(self):
        input_df = pandas.DataFrame({0: [1 for value in range(10)],
                                     1: [value for value in range(10)],
                                     # generate 2 fog-events
                                     23: ([2000 for value in range(3)] + [1900 for value in range(2)]
                                          + [2000 for value in range(2)] + [1900 for value in range(3)])})
        output_df = src.features.make_features.identify_log_events(input_df)
        self.assertTrue(['fog', 'time'] not in output_df.columns.tolist())

class TestExtractSequences(unittest.TestCase):
    def setUp(self) -> None:
        self.fog_df = pandas.DataFrame({0: numpy.random.randint(10, size=100),
                                        1: numpy.random.randint(20, size=100)})
        self.input_df = self.fog_df[20:25]

    def test_that_output_sequence_has_window_lenght(self):

        output_df = src.features.make_features.extract_sequences(self.input_df, self.fog_df, window_length=11,
                                                                 id_generator=src.features.make_features.id_generator())
        self.assertEqual(len(output_df), 11)

    def test_that_output_dataframe_has_non_empty_values_when_input_has_non_empty_values(self):
        # self.input_df has values before
        output_df = src.features.make_features.extract_sequences(self.input_df, self.fog_df, window_length=11,
                                                                 id_generator=src.features.make_features.id_generator())
        # check if out_df contains nans
        self.assertFalse(output_df.loc[:, [0, 1]].isnull().values.any())


    def test_that_output_dataframe_has_empty_values_when_input_data_has_empty_values(self):
        # extract intentionally empty
        input_df = self.fog_df.iloc[5:25]
        output_df = src.features.make_features.extract_sequences(input_df, self.fog_df, window_length=11,
                                                                 id_generator=src.features.make_features.id_generator())
        # check if out_df contains nans
        self.assertTrue(output_df.loc[:, [0, 1]].isnull().values.any())

    def test_that_output_contains_id_values(self):
        output_df = src.features.make_features.extract_sequences(self.input_df, self.fog_df, window_length=11,
                                                                 id_generator=src.features.make_features.id_generator())
        self.assertEqual(output_df['id'].unique().tolist(), [0])

    def test_that_output_time_values_has_same_lenght_as_window_lenght(self):
        output_df = src.features.make_features.extract_sequences(self.input_df, self.fog_df, window_length=10,
                                                                 id_generator=src.features.make_features.id_generator())
        self.assertEqual(output_df['time'].unique().tolist(), list(range(10)))

class TestExtractLabels(unittest.TestCase):
    def test_that_output_contains_the_minimum_value(self):
        input_df = pandas.DataFrame({23: numpy.random.randint(0, 10, size=10) + numpy.array(-10)})
        out = src.features.make_features.extract_labels(input_df)
        self.assertEqual(out, -10)

    
if __name__ == '__main__':
    unittest.main()
