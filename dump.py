import numpy as np
import matplotlib.pyplot as plt
import json
import mobility_data as md
import pandas as pd
import charging as ch


def read_json_config(keyword):
    with open('relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns[keyword]


# create the file path taken from json config
folder = read_json_config('folder')
file_name = 'quarterly_simulation_80.csv'
path = folder + '/' + file_name

# read mobility data
mobility_data = pd.read_csv(path)

# prepare the mobility data and aggregate
mobility_data = md.prepare_mobility_data(df=mobility_data,
                                         starting_date='2008-07-12 00:00:00',
                                         days=1)

mobility_data_aggregated = md.aggregate_15_min_steps(mobility_data)

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
mobility_data_aggregated['CHARGING'] = mobility_data_aggregated.apply(lambda row: charging(row['ECONSUMPTIONKWH'], row['previous_value']), axis=1)
# print(mobility_data_aggregated['CHARGING'])


def calc_power_in_battery(consumption, charging_power, charging):
    max_power = 50
    current_power = max_power
    if charging:
        current_power += charging_power
        if current_power > max_power:
            current_power = 50
    else:
        current_power -= consumption
        if current_power < 0:
            current_power = 0

        # print(current_power)

# mobility_data_aggregated['BATTERY_POWER'] = mobility_data_aggregated.apply(
#     lambda row: calc_power_in_battery(
#         consumption=row['ECONSUMPTIONKWH'],
#         start_power=50
#     )
# )

class ElectricVehicle:
    def __init__(self, name, battery_size):
        self.name = name
        self.battery_size = battery_size
        self.battery_level = 0

    def charge_battery(self):
        self.battery_level = 100

    def drive(self, km):
        consumption = km * 1
        if self.battery_level == 0:
            print("Battery is dead. Please charge the vehicle.")
            return
        self.battery_level -= consumption
        print(self.battery_level)

# Create an instance of the ElectricVehicle class
my_ev = ElectricVehicle('Tesla', 50)
for days in range(0, 10, 1):
    my_ev.charge_battery()
    my_ev.drive(25)
    