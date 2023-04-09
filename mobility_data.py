import pandas as pd
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
import helper as helper


class DataAggregator:
    def __init__(self):
        self.df_limited_time = pd.DataFrame()
        self.df_processed = pd.DataFrame()

        self.starting_date = None
        self.days = None
        self.end_date = None

    def __create_df_limited_time(self, df: pd.DataFrame, starting_date: str, days: int):
        self.days = days
        self.starting_date = pd.to_datetime(starting_date)
        self.end_date = self.starting_date + timedelta(days=days)
        df.loc[:, 'TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        self.df_limited_time = df[(df['TIMESTAMP'] >= starting_date) & (df['TIMESTAMP'] < self.end_date)]
        # return self.df_limited_time

    def __aggregate_15_min_steps(self):
        # calculate the total energy demand in that 15 minutes
        self.df_limited_time.set_index('TIMESTAMP', inplace=True)

        def __aggregation_mode():
            """Helper function to find the mode"""
            return lambda x: x.value_counts().index[0]

        # Refactor, columns are already in json relevant columns
        self.df_processed = self.df_limited_time.resample('15T', closed='left').agg(
            {'ECONSUMPTIONKWH': 'sum',
             'TRIPNUMBER': 'min',
             'LONGITUDE': 'first',
             'LATITUDE': 'first',
             'ID_PANELSESSION': 'max',
             'SPEED': 'mean',
             'ID_TERMINAL': __aggregation_mode(),
             'DELTATIME': 'sum',
             'DELTAPOS': 'sum',
             'ID_LOCATIONTYPE': __aggregation_mode(),
             'CLUSTER': __aggregation_mode(),
             'ORIGINAL': __aggregation_mode()}
        )
        # return self.df_processed

    def __data_cleaning(self):
        # Every row needs an entry in tripno, econ
        self.df_processed = self.df_processed.dropna(subset=['TRIPNUMBER', 'ECONSUMPTIONKWH'])
        # All timestamps need to match the given format
        timestamp_format = '%Y-%m-%d %H:%M'
        try:
            pd.to_datetime(self.df_processed.index, format=timestamp_format, errors='coerce').notnull().all()
            print("index in correct format.")
        except ValueError:
            print("Timestamp error.")
        if self.df_processed.shape[0] % 96 == 0:
            print('Number of rows can be divided by 96.')
        else:
            raise Exception('Number of rows cannot be divided by 96.')
        #TODO
        # Could also cast every column used in simulation to specific dtype, that we do not encounter errors
        # during simulation

    def prepare_mobility_data(self, df: pd.DataFrame, starting_date: str, days: int):
        self.__create_df_limited_time(df, starting_date, days)
        self.__aggregate_15_min_steps()
        self.__data_cleaning()


def turn_into_dict(df: pd.DataFrame):
    mobility_dict = df.T.to_dict('dict')
    return mobility_dict


if __name__ == '__main__':
    path1 = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    path2 = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    try:
        mobility_data = pd.read_csv(path1)
    except FileNotFoundError:
        mobility_data = pd.read_csv(path2)

    data = DataAggregator()
    data.prepare_mobility_data(mobility_data, '2008-07-13 00:00:00', 1)
    print(data.df_processed)
    # df = data.create_df_limited_time()
    # df_1 = data.aggregate_15_min_steps()
    # data.data_cleaning()

    # mobility_data = prepare_mobility_data(df=mobility_data,
    #                                       starting_date='2008-07-12 00:00:00',
    #                                       days=1)
    #
    # mobility_data_aggregated = aggregate_15_min_steps(mobility_data)
    #
    # mobility_data_aggregated = turn_into_dict(mobility_data_aggregated)
    # key = pd.to_datetime('2008-07-12 01:15:00')
    # print(mobility_data_aggregated[key]['ECONSUMPTIONKWH'])
