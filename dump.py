import numpy as np
import matplotlib.pyplot as plt
import json
import mobility_data as md
import pandas as pd
import charging as ch
import math
from car_agent import ElectricVehicle


def read_json_config(keyword):
    with open('relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns[keyword]


def create_file_path():
    # create the file path taken from json config
    folder = read_json_config('folder')
    file_name = 'quarterly_simulation_80.csv'
    path = folder + '/' + file_name
    return path


# read mobility data
mobility_data = pd.read_csv(create_file_path())

# prepare the mobility data and aggregate
mobility_data = md.prepare_mobility_data(df=mobility_data,
                                         starting_date='2008-07-12 00:00:00',
                                         days=1)

mobility_data_aggregated = md.aggregate_15_min_steps(mobility_data)
# print(mobility_data_aggregated)


class MobilityData:
    def __init__(self, header, row):
        self.__dict__ = dict(zip(header, row))


test = dict(zip(mobility_data_aggregated.columns, mobility_data_aggregated[1:]))

instances = MobilityData(mobility_data_aggregated.columns, mobility_data_aggregated)

print(test)

breakpoint()
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


#
mobility_data_aggregated['CHARGING'] = mobility_data_aggregated.apply(
    lambda row: charging(row['ECONSUMPTIONKWH'], row['previous_value']), axis=1)


# print(mobility_data_aggregated['CHARGING'])


class ElectricVehicle2:
    def __init__(self, name, battery_capacity):
        self.name = name
        self.battery_capacity = battery_capacity
        self.stop_charging = battery_capacity * 0.8
        self.battery_power_left = None
        self.charging_power = 10
        self.moving = False
        self.consumption = None
        self.charging_buffer = True

    def __charging_possible(self):
        if self.consumption == 0:
            if self.charging_buffer:
                self.moving = True  # charging not possible because of buffer therefore moving
                self.charging_buffer = False
            else:
                self.moving = False
        else:
            self.moving = True
            self.charging_buffer = True

    def calc_power_in_battery(self):
        # check if charging is possible
        self.__charging_possible()
        # initial set battery power left to the maximum capacity
        if self.battery_power_left is None:
            self.battery_power_left = self.battery_capacity
        # if moving subtract consumption
        if self.moving:
            self.battery_power_left -= self.consumption
            if self.battery_power_left < 0:
                self.battery_power_left = 0
        else:
            self.battery_power_left += self.charging_power
            if self.battery_power_left > self.stop_charging:
                self.battery_power_left = self.stop_charging


def generate_cars_according_to_dist(number_of_agents):
    with open('car_values.json', 'r') as f:
        data = json.load(f)

    total_cars = 0
    for name in data.keys():
        total_cars += data[name]["number"]

    cars = []
    distribution = []
    for name in data.keys():
        cars += [name]
        distribution += [data[name]["number"] / total_cars]

    car_names = np.random.choice(cars, size=number_of_agents, p=distribution)
    # print(len(car_names), "car names generated.")

    return car_names


def number_of_each_car(car_name, car_names_list):
    car_series = pd.Series(car_names_list)
    car_counts = car_series.value_counts()
    if car_name in car_counts.index:
        return car_counts[car_name]
    else:
        return 0

# car_models = generate_cars_according_to_dist(10)
# number = number_of_each_car(car_name='renault_zoe', car_names_list=car_models)
# print(number)
# create an agent like the distribution
# for model in car_models:
#     car_agent = ElectricVehicle(model)
#     normal_capacity = car_agent.get_battery_capacity('normal')
#     agents = generate_cars_according_to_dist(car_agent.number_of_car)

# print(number)
# print(car_agent.name, car_agent.number_of_car, normal_capacity)
