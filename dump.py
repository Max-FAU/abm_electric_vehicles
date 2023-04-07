import numpy as np
import matplotlib.pyplot as plt
import json
import mobility_data as md
import pandas as pd
import charging as ch
import math
from datetime import datetime
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


#
mobility_data_aggregated['CHARGING'] = mobility_data_aggregated.apply(
    lambda row: charging(row['ECONSUMPTIONKWH'], row['previous_value']), axis=1)


# print(mobility_data_aggregated['CHARGING'])


class ElectricVehicle:
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


zoe = ElectricVehicle('renault_zoe', 110)


def power(battery_power, consumption, charging):
    if battery_power is None:
        battery_power = max_capacity
    else:
        if consumption > 0:
            battery_power -= consumption
            if battery_power < 0:
                battery_power = 0
        elif consumption == 0:
            battery_power += charging
            if battery_power >= max_capacity:
                battery_power = max_capacity
        else:
            battery_power = battery_power
            print("negative consumption not possible")
    print(battery_power)
    return battery_power

data_tracking_df = pd.DataFrame(columns=['timestep', 'battery_power', 'consumption'])
max_capacity = zoe.battery_capacity
battery_power = None
charging_power = 10
for timestep, data in mobility_data_aggregated.iterrows():
    # print(data['ECONSUMPTIONKWH'])
    battery_power = power(battery_power, data['ECONSUMPTIONKWH'], charging_power)
    data_tracking_df = data_tracking_df.append({"timestep": timestep, "battery_power": battery_power, "consumption": data['ECONSUMPTIONKWH']}, ignore_index=True)



print(data_tracking_df)
# data_tracking_df.to_csv('test.csv')
# print(number)
# print(car_agent.name, car_agent.number_of_car, normal_capacity)

# create plot with two y-axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# plot y1 on first y-axis
ax1.plot(data_tracking_df['timestep'], data_tracking_df['battery_power'], 'b-', label='battery capacity')
ax1.set_xlabel('Time')
ax1.set_ylabel('battery capacity in kwh')
ax1.tick_params('y', colors='b')

# plot y2 on second y-axis
ax2.plot(data_tracking_df['timestep'], data_tracking_df['consumption'], 'r-', label='consumption')
ax2.set_ylabel('consumption in kwh')
ax2.tick_params('y', colors='r')

# set xticks and rotate labels
ax1.set_xticks(data_tracking_df['timestep'][::4])
xticklabels = [datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H:%M') for x in data_tracking_df['timestep'][::4]]
ax1.set_xticklabels(xticklabels, rotation=90)

# add legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# display plot
plt.tight_layout()
plt.show()