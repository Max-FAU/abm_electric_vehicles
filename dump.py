import numpy as np
from mesa import Agent
import json
import mobility_data as md
import pandas as pd
import charging as ch
import math
from itertools import count
import matplotlib.pyplot as plt
import random
import helper as helper
from mobility_data import MobilityDataAggregator
from car_agent import ElectricVehicle


# from car_agent import ElectricVehicle

# print(mobility_data_aggregated)

# # calculate the absolute battery level using cumsum starting from 50
# mobility_data_aggregated['BATTERY_LEVEL'] = 50 - mobility_data_aggregated['ECONSUMPTIONKWH'].cumsum()
#

# grid.set_up_grid()

# for timestep, data in mobility_data_aggregated.iterrows():
#     battery_power = dummy.power(consumption=data['ECONSUMPTIONKWH'])
#
# print(dummy.load_curve)
# print(dummy.battery_level_curve)
# data_tracking_df.to_csv('test.csv')
# print(number)
# print(car_agent.name, car_agent.number_of_car, normal_capacity)
#
# df = pd.read_csv('aggregated_results.csv')
# df.set_index('timestamp', inplace=True)
#
# fig, ax = plt.subplots()
# ax.plot(df)
# ax.set_xticks(df.index[::4])
# ax.tick_params(axis='x', labelrotation=90)
#
# plt.tight_layout()
# plt.show()

def calculate_range_anxiety():
    pass


def read_data_local_test():
    path1 = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_295.csv"
    path2 = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    try:
        mobility_data = pd.read_csv(path1)
    except FileNotFoundError:
        mobility_data = pd.read_csv(path2)
    return mobility_data


def load_test_timestamps():
    start_time = pd.Timestamp('2008-07-17 12:00:00')
    timestamps = pd.date_range(start_time, periods=24, freq='15min').tolist()
    return timestamps


if __name__ == '__main__':
    # timestamps = load_test_timestamps()
    # mobility_data = read_data_local_test()
    #
    # bmw_i3 = ElectricVehicle(model='bmw_i3', target_soc=1.00)
    #
    # data = MobilityDataAggregator(mobility_data)
    # mobility_data = data.prepare_mobility_data(start_date='2008-07-13', num_days=7)
    # mobility_dict = mobility_data.T.to_dict('dict')
    # # print(mobility_dict)
    #
    # for step in timestamps:
    #     timestamp_consumption = mobility_dict[step]['ECONSUMPTIONKWH']
    #     bmw_i3.calculate_battery_level(consumption=timestamp_consumption,
    #                                    charging_efficiency=0.95)
    #     print(bmw_i3.battery_level)
    #
    # calculate_range_anxiety()
    with open('private_cars.json', 'r') as f:
        data = json.load(f)
    private_car_ids = [d['id'] for d in data] # extract the ids
    print(private_car_ids)
