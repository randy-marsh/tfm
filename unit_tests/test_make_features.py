import unittest
import pandas
import numpy
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


if __name__ == '__main__':
    unittest.main()
