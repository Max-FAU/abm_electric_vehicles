import numpy as np
from mesa import Agent
import json
import mobility_data as md
import pandas as pd
import charging as ch
import math
from itertools import count

import helper as helper

# from car_agent import ElectricVehicle


filepath = helper.read_json_config('file')
# read mobility data
mobility_data = pd.read_csv(filepath)

# prepare the mobility data and aggregate
mobility_data = md.prepare_mobility_data(df=mobility_data,
                                         starting_date='2008-07-12 00:00:00',
                                         days=1)

mobility_data_aggregated = md.aggregate_15_min_steps(mobility_data)
# print(mobility_data_aggregated)

# # calculate the absolute battery level using cumsum starting from 50
# mobility_data_aggregated['BATTERY_LEVEL'] = 50 - mobility_data_aggregated['ECONSUMPTIONKWH'].cumsum()
#

mobility_data_aggregated['previous_value'] = mobility_data_aggregated['ECONSUMPTIONKWH'].shift(1)


def charging(econ_value, prev_econ_value):
    if econ_value == 0 and prev_econ_value == 0:
        # print('charging possible')
        charging = True
    # elif econ_value == 0 and prev_econ_value is None:
    #     print('This is the first step, wait one more step')
    #     charging = False
    # elif econ_value == 0 and prev_econ_value > 0:
    #     print('Wait one more step to start charging')
    #     charging = False
    else:
        # print('charging not possible')
        charging = False
    return charging


class ElectricityGrid:
    number_cars = 0

    def __init__(self, car):
        self.car = car
        ElectricityGrid.number_cars += 1

#     def distribute_cars_to_bus(self, cars_per_bus):
#         busses = car[i:i+8] for i in range(0, len(car), 8)
#
# grid = ElectricityGrid(1)
# print(grid2.id)

#
mobility_data_aggregated['CHARGING'] = mobility_data_aggregated.apply(
    lambda row: charging(row['ECONSUMPTIONKWH'], row['previous_value']), axis=1)

# print(mobility_data_aggregated['CHARGING'])

# for timestep, data in mobility_data_aggregated.iterrows():
#     battery_power = dummy.power(consumption=data['ECONSUMPTIONKWH'])
#
# print(dummy.load_curve)
# print(dummy.battery_level_curve)
# data_tracking_df.to_csv('test.csv')
# print(number)
# print(car_agent.name, car_agent.number_of_car, normal_capacity)
