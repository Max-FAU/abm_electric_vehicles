import pandas as pd
from datetime import timedelta
import json
import matplotlib.pyplot as plt
import numpy as np


def create_df_limited_time(df: pd.DataFrame, starting_date: str, days: int):
    start_date = pd.to_datetime(starting_date)
    end_date = start_date + timedelta(days=days)
    df.loc[:, 'TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df[(df['TIMESTAMP'] > start_date) & (df['TIMESTAMP'] < end_date)]

    return df


def aggregate_15_min_steps(df: pd.DataFrame):
    # calculate the total energy demand in that 15 minutes
    df.set_index('TIMESTAMP', inplace=True)

    def aggregation_mode():
        """Helper function to find the mode"""
        return lambda x: x.value_counts().index[0]

    # Refactor, columns are already in json relevant columns
    df_resampled = df.resample('15T').agg(
        {'ECONSUMPTIONKWH': 'sum',
         'TRIPNUMBER': 'min',
         'LONGITUDE': 'first',
         'LATITUDE': 'first',
         'ID_PANELSESSION': 'max',
         'SPEED': 'mean',
         'ID_TERMINAL': aggregation_mode(),
         'DELTATIME': 'sum',
         'DELTAPOS': 'sum',
         'ID_LOCATIONTYPE': aggregation_mode(),
         'CLUSTER': aggregation_mode(),
         'ORIGINAL': aggregation_mode()}
    )
    return df_resampled


def read_json_relevant_columns():
    with open('relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns['relevant_columns']


def data_cleaning(df: pd.DataFrame):
    # Every row needs an entry in timestamp, tripno, econ
    df = df.dropna(subset=['TIMESTAMP', 'TRIPNUMBER', 'ECONSUMPTIONKWH'])
    # All timestamps need to match the given format
    timestamp_format = '%Y-%m-%d %H:%M'
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format=timestamp_format, errors='coerce')
    df.dropna(subset=['TIMESTAMP'], inplace=True)
    # Could also cast every column used in simulation to specific dtype, that we do not encounter errors
    # during simulation
    return df


def prepare_mobility_data(df: pd.DataFrame, starting_date: str, days: int):
    relevant_cols = read_json_relevant_columns()
    df = df[relevant_cols]
    df = create_df_limited_time(df=df,
                                starting_date=starting_date,
                                days=days)
    df = data_cleaning(df)
    return df


if __name__ == '__main__':
    path1 = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    path2 = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    try:
        mobility_data = pd.read_csv(path1)
    except FileNotFoundError:
        mobility_data = pd.read_csv(path2)

    mobility_data = prepare_mobility_data(df=mobility_data,
                                          starting_date='2008-07-12 00:00:00',
                                          days=14)

    mobility_data_aggregated = aggregate_15_min_steps(mobility_data)

    plot = False
    if plot:
        try:
            test = pd.read_csv(path1)
        except FileNotFoundError:
            test = pd.read_csv(path2)

        test = prepare_mobility_data(df=test,
                                     starting_date='2008-07-12 00:00:00',
                                     days=14)

        test.set_index('TIMESTAMP', inplace=True)

        charging_1 = (test['ECONSUMPTIONKWH'] <= 0)
        charging_2 = (mobility_data_aggregated['ECONSUMPTIONKWH'] <= 0)


        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

        ax1.plot(test['ECONSUMPTIONKWH'], color='blue')
        ax1.set_ylabel('baseline_value')

        ax1.fill_between(test.index,
                         np.min(mobility_data_aggregated['ECONSUMPTIONKWH']),
                         np.max(mobility_data_aggregated['ECONSUMPTIONKWH']) * 1.1,
                         where=charging_1, alpha=0.3, color='green')


        ax2.plot(mobility_data_aggregated['ECONSUMPTIONKWH'], color='black')
        ax2.set_ylabel('aggregated_value')

        ax2.fill_between(mobility_data_aggregated.index,
                         np.min(mobility_data_aggregated['ECONSUMPTIONKWH']),
                         np.max(mobility_data_aggregated['ECONSUMPTIONKWH']) * 1.1,
                         where=charging_2, alpha=0.3, color='green')


        ylim_max_value = max(max(test['ECONSUMPTIONKWH']),
                             max(mobility_data_aggregated['ECONSUMPTIONKWH'])) * 1.1

        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', labelrotation=90)
            ax.set_ylim(0, ylim_max_value)


        plt.tight_layout()
        plt.show()

    print(mobility_data_aggregated)