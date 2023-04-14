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
df = pd.read_csv('median_trip_length.csv', index_col=0)
df = df.dropna()
df['decile_label'] = pd.qcut(df['median_trip_length'], q=10, labels=False, duplicates='drop') + 1
df = df.sort_values('decile_label')
df = df.reset_index(drop=True)

with open('car_values.json') as f:
    car_dict = json.load(f)

charging_power_list = [car["battery_capacity"] for car in car_dict.values()]
charging_power_list = sorted(charging_power_list)

print(charging_power_list)


# TODO
# Directory with all mobility data
# Mobility data labelled with 
# list of
mobility_files = ["All file names"]
car_id = 20
mobility_car_id = '_' + str(car_id)
valid_file_names = [name for name in mobility_files if mobility_car_id in name]
picked = random.choice(mobility_files)
mobility_files.remove(picked)

print("Picked number:", picked)
print("Remaining list:", my_list)