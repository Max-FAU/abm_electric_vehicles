import pandas as pd
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
import helper as helper


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
            {'ECONSUMPTIONKWH': 'sum',
             'TRIPNUMBER': 'min',
             'LONGITUDE': 'first',
             'LATITUDE': 'first',
             'ID_PANELSESSION': 'max',  # charging should be dependent on this / consumption is doing it
             'SPEED': 'mean',
             'ID_TERMINAL': _aggregation_mode(),  # TODO set to first is always the same
             'DELTATIME': 'sum',
             'DELTAPOS': 'sum',
             'ID_LOCATIONTYPE': _aggregation_mode(),
             'CLUSTER': 'min',   # home / work location
             'ORIGINAL': _aggregation_mode()}
        )
        # return self.df_processed

    def _data_cleaning(self):
        # Every row needs an entry in tripno, econ
        self.df_processed = self.df_processed.dropna(subset=['TRIPNUMBER', 'ECONSUMPTIONKWH'])
        # All timestamps need to match the given format
        timestamp_format = '%Y-%m-%d %H:%M'
        try:
            pd.to_datetime(self.df_processed.index, format=timestamp_format, errors='coerce').notnull().all()
        except ValueError:
            print("Timestamp index error - it is in wrong format.")
        if not self.df_processed.shape[0] % 96 == 0:
            raise Exception('Number of rows cannot be divided by 96.')
        #TODO
        # Could also cast every column used in simulation to specific dtype, that we do not encounter errors
        # during simulation

    def prepare_mobility_data(self, starting_date: str, num_days: int) -> pd.DataFrame:
        self._set_unique_id()
        self._create_df_limited_time(starting_date, num_days)
        self._aggregate_15_min_steps()
        self._data_cleaning()
        return self.df_processed


if __name__ == '__main__':
    path1 = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    path2 = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    try:
        mobility_data = pd.read_csv(path1)
    except FileNotFoundError:
        mobility_data = pd.read_csv(path2)

    data = MobilityDataAggregator(mobility_data)
    mobility_data = data.prepare_mobility_data(starting_date='2008-07-13', num_days=1)
    # mobility_dict = mobility_data.T.to_dict('dict')
    print(data.unique_id)