import pandas as pd
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
import helper as helper
import glob
import timeit
import sys

class MobilityDataAggregator:
    def __init__(self, raw_mobility_data):
        self.raw_mobility_data = raw_mobility_data
        self.df_limited_time = pd.DataFrame()
        self.df_processed = pd.DataFrame()

        self.unique_id = None
        self.starting_date = None
        self.days = None
        self.end_date = None

    def _set_unique_id(self, id_col='ID_TERMINAL'):
        unique_id = self.raw_mobility_data[id_col].unique()
        if len(unique_id) > 1:
            raise Exception('More than one IDs in id_col.')
        self.unique_id = unique_id[0]

    def _create_df_limited_time(self, starting_date: str, days: int):
        self.days = days
        self.starting_date = pd.to_datetime(starting_date)
        self.end_date = self.starting_date + timedelta(days=days)
        self.raw_mobility_data.loc[:, 'TIMESTAMP'] = pd.to_datetime(self.raw_mobility_data['TIMESTAMP'])
        self.df_limited_time = self.raw_mobility_data[(self.raw_mobility_data['TIMESTAMP'] >= starting_date) & (self.raw_mobility_data['TIMESTAMP'] < self.end_date)]
        # return self.df_limited_time

    def _aggregate_15_min_steps(self):
        # calculate the total energy demand in that 15 minutes
        self.df_limited_time.set_index('TIMESTAMP', inplace=True)

        def _aggregation_mode():
            """Helper function to find the mode"""
            return lambda x: x.value_counts().index[0]

        # Refactor, columns are already in json relevant columns
        self.df_processed = self.df_limited_time.resample('15T', closed='left').agg(
            {'ECONSUMPTIONKWH': 'sum',  # charging dependent on this
             'TRIPNUMBER': 'min',
             'ID_PANELSESSION': 'max',  # charging dependent on this
             'ID_TERMINAL': 'first',  # car_id
             'CLUSTER': 'min',
             'DELTAPOS': 'sum'}   # home / work location
        )
        # return self.df_processed
        self.df_processed = self.df_processed.ffill()  # fill missing values with the following value

    def _data_cleaning(self):
        try:
            # Every row needs an entry in tripno, econ
            self.df_processed = self.df_processed.dropna(subset=['TRIPNUMBER', 'ECONSUMPTIONKWH'])
        except ValueError:
            print("Dropping NAN values in dataframe failed.")
        # All timestamps need to match the given format
        timestamp_format = '%Y-%m-%d %H:%M'
        try:
            pd.to_datetime(self.df_processed.index, format=timestamp_format, errors='coerce').notnull().all()
        except ValueError:
            print("Timestamp index error - it is in wrong format.")
        if not self.df_processed.shape[0] % 96 == 0:
            file_name = str(self.unique_id) + '_failed_processed_df.csv'  # save the failed df
            self.df_processed.to_csv(file_name)
            print('Number of rows cannot be divided by 96.')

    def prepare_mobility_data(self, starting_date: str, num_days: int) -> pd.DataFrame:
        self._set_unique_id()
        self._create_df_limited_time(starting_date, num_days)
        self._aggregate_15_min_steps()
        self._data_cleaning()
        return self.df_processed


def median_trip_length(df, car_id):
    len_dict = {}
    trip_df = df.groupby('TRIPNUMBER').sum()
    med_trip_len = trip_df['DELTAPOS'].median()
    len_dict[car_id] = med_trip_len
    print('Successfully added entry for car {} to len dict.'.format(car_id))
    return len_dict


if __name__ == '__main__':
    start = timeit.default_timer()
    # define the directory path
    directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
    csv_files = glob.glob(directory_path + "/*906.csv")

    id_segmentation = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\segmented_ids_07112021.xlsx"
    id_segmentation_df = pd.read_excel(id_segmentation)
    id_segmentation_df = id_segmentation_df[['CLUSTER', 'id']]

    # Open a file for writing
    with open('log.txt', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f

        len_dict = {}
        for file in csv_files:
            time_loop_iteration_start = timeit.default_timer()
            mobility_data = pd.read_csv(file)
            unique_id = mobility_data['ID_TERMINAL'].unique()[0]
            if id_segmentation_df.loc[id_segmentation_df['id'] == unique_id, 'CLUSTER'].values[0] in [1, 4, 5, 7]:
                data = MobilityDataAggregator(mobility_data)
                mobility_data = data.prepare_mobility_data(starting_date='2008-07-13', num_days=7)
                new_dict = median_trip_length(mobility_data, unique_id)
                len_dict.update(new_dict)
            time_loop_iteration_end = timeit.default_timer()
            print('Time: ', time_loop_iteration_end - time_loop_iteration_start, ' secs.')

        # print(len_dict)
        sorted_dict = dict(sorted(len_dict.items(), key=lambda item: item[1]))

        sorted_dict_df = pd.DataFrame(sorted_dict, index=[0])
        sorted_dict_df = sorted_dict_df.T
        sorted_dict_df = sorted_dict_df.reset_index()
        sorted_dict_df.columns = ['car_id', 'median_trip_length']
        sorted_dict_df.to_csv("median_trip_length.csv")
        stop = timeit.default_timer()
        print('Total Runtime: ', stop - start)

    # Reset stdout to its default value
    sys.stdout = sys.__stdout__

    # path1 = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_295.csv"
    # path2 = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    # try:
    #     mobility_data = pd.read_csv(path1)
    # except FileNotFoundError:
    #     mobility_data = pd.read_csv(path2)
    # print(mobility_data)
    #
    # data = MobilityDataAggregator(mobility_data)
    # mobility_data = data.prepare_mobility_data(starting_date='2008-07-13', num_days=7)
    # print(mobility_data)

    # mobility_dict = mobility_data.T.to_dict('dict')


