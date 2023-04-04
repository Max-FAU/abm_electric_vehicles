import random

import mesa
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class ElectricVehicle(mesa.Agent):
    def __init__(self, unique_id, battery_capacity, current_consumption, power_left_in_battery, status, charging_power, model):
        super().__init__(unique_id, model)
        self.battery_capacity = battery_capacity
        self.current_consumption = current_consumption
        self.power_left_in_battery = power_left_in_battery - current_consumption
        self.soc_battery = power_left_in_battery / battery_capacity
        self.status = status
        self.charging_power = charging_power

    def step(self):

        self.power_left_in_battery = self.power_left_in_battery - self.current_consumption

        if self.power_left_in_battery <= 0.2 and self.status == 'driving':
            self.status = 'charging'

        if self.power_left_in_battery == 0:
            print("My Battery is empty for agent " + str(self.unique_id) + ".")
            return

        if self.status == 'charging':
            self.power_left_in_battery += self.charging_power


class ChargingModel(mesa.Model):
    def __init__(self, N):
        self.number_evs = N
        self.schedule = mesa.time.RandomActivation(self)
        # create agents
        for i in range(self.number_evs):
            a = ElectricVehicle(i, self)
            self.schedule.add(a)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()


if __name__ == "__main__":
    soc_all_cars = []
    # This runs the model 100 times, each model executing 10 steps.
    for j in range(1):
        # Run the model with 5 cars / agents
        model = ChargingModel(5)

        # 20 trips or steps for the model
        for i in range(20):
            model.step()

    #     # Store the results
    #     for agent in model.schedule.agents:
    #         soc_all_cars.append(agent.soc_battery)
    #
    # plt.hist(soc_all_cars, bins=range(max(soc_all_cars) + 1))
    # plt.show()