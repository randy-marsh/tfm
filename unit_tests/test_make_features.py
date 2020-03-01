import unittest
import pandas
import numpy
import random
import src.features.make_features


class TestIdentifyFogEvents(unittest.TestCase):
    def test_that_if_input_has_one_long_fog_event_then_output_has_one_group(self):
        input_df = pandas.DataFrame({
                                     # generate 1 fog-events
                                     'rvr': [1 for value in range(10)]})
        output_df = src.features.make_features.identify_log_events(input_df)
        groups = output_df['groups'].unique()
        groups_len = len(groups[~numpy.isnan(groups)])
        self.assertEqual(groups_len, 1)

    def test_that_if_input_has_two_fog_events_separated_by_one_hour_then_output_has_two_groups(self):
        input_df = pandas.DataFrame({
                                     # generate 2 fog-events
                                     'rvr': ([1900 for value in range(5)] + [2000 for value in range(1)]
                                          + [1900 for value in range(4)])})
        output_df = src.features.make_features.identify_log_events(input_df)
        groups = output_df['groups'].unique()
        groups_len = len(groups[~numpy.isnan(groups)])
        self.assertEqual(groups_len, 2)

    def test_that_if_input_df_has_two_fog_events_separated_by_more_than_one_hour_then_output_has_two_groups(self):
        input_df = pandas.DataFrame({
                                     # generate 2 fog-events
                                     'rvr': ([2000 for value in range(3)] + [1900 for value in range(2)]
                                          + [2000 for value in range(2)] + [1900 for value in range(3)])})
        output_df = src.features.make_features.identify_log_events(input_df)
        groups = output_df['groups'].unique()
        groups_len = len(groups[~numpy.isnan(groups)])
        self.assertEqual(groups_len, 2)

    def test_that_output_not_contain_fog_or_time_columns(self):
        input_df = pandas.DataFrame({
                                     # generate 2 fog-events
                                     'rvr': ([2000 for value in range(3)] + [1900 for value in range(2)]
                                          + [2000 for value in range(2)] + [1900 for value in range(3)])})
        output_df = src.features.make_features.identify_log_events(input_df)
        self.assertTrue(['fog', 'time'] not in output_df.columns.tolist())


class TestExtractSequences(unittest.TestCase):
    def setUp(self) -> None:
        self.fog_df = pandas.DataFrame({0: numpy.random.randint(10, size=100),
                                        1: numpy.random.randint(20, size=100)})
        self.input_df = self.fog_df.iloc[20:25]

    def test_that_output_dataframe_is_as_subset_from_fog_events_df(self):
        output_df = src.features.make_features.extract_sequences(self.input_df, self.fog_df, window_length=11,
                                                                 id_generator=src.features.make_features.id_generator())

        self.assertTrue(output_df.loc[:, [0, 1]].equals(self.fog_df.iloc[9:20]))

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
        input_df = pandas.DataFrame({23: numpy.random.randint(0, 10, size=10).tolist() + [-100]})
        out = src.features.make_features.extract_labels(input_df)
        self.assertEqual(out, -100)

    def test_that_if_there_are_two_minimum_then_only_returns_one(self):
        input_df = pandas.DataFrame({23: numpy.random.randint(0, 10, size=10).tolist() + [-100., -100.]})
        out = src.features.make_features.extract_labels(input_df)
        with self.assertRaises(TypeError):
            len(out)


class TestVRVFirstFogEvent(unittest.TestCase):
    def test_that_if_minimun_is_first_then_returns_false(self):
        input_df = pandas.DataFrame({23: [1000, 1999, 2000, 1999, 1999]})
        self.assertFalse(src.features.make_features.is_vrv_min_at_first_fog_event(input_df))

    def test_that_if_minimun_is_not_first_then_returns_True(self):
        input_df = pandas.DataFrame({23: [2000, 1000, 2000, 2000, 2000]})
        self.assertTrue(src.features.make_features.is_vrv_min_at_first_fog_event(input_df))


class TestGetFogEvents(unittest.TestCase):
    def setUp(self) -> None:
        self.input_df = pandas.DataFrame({'groups': [numpy.nan, 0, 0, numpy.nan, 1]})

    def test_that_if_input_has_nan_then_output_groups_does_not_have_nan(self):
        out_list = src.features.make_features.get_fog_events(self.input_df)
        self.assertFalse(numpy.isnan(out_list).any())

    def test_that_output_has_all_input_groups(self):
        out_list = src.features.make_features.get_fog_events(self.input_df)
        self.assertEqual([0, 1], out_list)

    def test_that_output_values_are_not_duplicated(self):
        out_list = src.features.make_features.get_fog_events(self.input_df)
        self.assertEqual(list(set(out_list)), [0, 1])


class TestIdGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = src.features.make_features.id_generator()

    def test_that_generator_start_in_zero(self):
        self.assertTrue(next(self.generator) == 0)

    def test_that_generator_increases_by_one(self):
        # obtain another value greater than zero
        next(self.generator)
        first_value = next(self.generator)
        last_value = next(self.generator)
        self.assertTrue((last_value - first_value) == 1)


class TestValidFogEventsGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.input_df = pandas.DataFrame({23: [1900, 1000] + [1000, 1900, 1900] + [1900 for value in range(4)] + [1000],
                                         'groups': [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]})

    def test_that_output_groups_are_valid(self):
        groups = list()
        for group in src.features.make_features.valid_fog_events(self.input_df, min_fog_event_length=3,
                                                                 max_fog_event_length=5, first_vrv=False):
            groups.append(group)
        self.assertEqual([1000, 1900, 1900], groups[0][23].tolist())

    def test_that_output_contains_groups_greater_than_min_fog_event_length(self):
        groups = list()
        for group in src.features.make_features.valid_fog_events(self.input_df, min_fog_event_length=3,
                                                                 max_fog_event_length=None, first_vrv=False):

            groups.append(group)
        self.assertEqual(len(groups), 2)

    def test_that_output_not_contains_groups_greater_than_max_fog_event_length_when_max_param_is_not_none(self):
        groups = list()
        for group in src.features.make_features.valid_fog_events(self.input_df, min_fog_event_length=3,
                                                                 max_fog_event_length=5, first_vrv=False):
            groups.append(group)
        self.assertEqual(len(groups), 1)

    def test_that_if_first_rvr_is_true_then_there_are_no_groups_starts_with_vrv_min(self):
        groups = list()
        for group in src.features.make_features.valid_fog_events(self.input_df, min_fog_event_length=3,
                                                                 max_fog_event_length=None, first_vrv=True):
            groups.append(group)
        self.assertEqual(len(groups), 1)


class TestGetFirstRVR(unittest.TestCase):
    def test_that_returns_first_rvr_value(self):
        input_df = pandas.DataFrame({23: [1234, 1954, 1954, 1954]})
        out = src.features.make_features.get_first_rvr(input_df, "Grupos_totales_continua")
        self.assertEqual(out, 1234)


if __name__ == '__main__':
    unittest.main()
