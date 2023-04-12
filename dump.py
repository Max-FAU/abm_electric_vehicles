import numpy as np
from mesa import Agent
import json
import mobility_data as md
import pandas as pd
import charging as ch
import math
from itertools import count
import matplotlib.pyplot as plt

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

df = pd.read_csv('aggregated_results.csv')
df.set_index('timestamp', inplace=True)

fig, ax = plt.subplots()
ax.plot(df)
ax.set_xticks(df.index[::4])
ax.tick_params(axis='x', labelrotation=90)

plt.tight_layout()
plt.show()
