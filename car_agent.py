# Initialization of the model
# Retrieve the EV Agent data
# create electric vehicle with battery
# assign battery capacity
# initialize electric vehicle
import json
import pandas as pd


class ElectricVehicle:
    def __init__(self, name, unique_id):
        self.name = name
        self.unique_id = unique_id
        self.battery_level = None
        self.small_battery_capacity = None
        self.normal_battery_capacity = None
        self.large_battery_capacity = None
        self.soc = None

        self._initialize_car_values()

        self.load_curve = []
        self.battery_level_curve = []
        self.soc_curve = []

    def _initialize_car_values(self):
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        self.small_battery_capacity = car_dict[self.name]["small_battery"]
        self.normal_battery_capacity = car_dict[self.name]["normal_battery"]
        self.large_battery_capacity = car_dict[self.name]["large_battery"]
        self.number_of_car = car_dict[self.name]["number"]
        self.charging_power = car_dict[self.name]["charging_power"]

    def calculate_battery_level(self, consumption, battery_capacity):
        if self.battery_level is None:
            self.battery_level = battery_capacity

        if consumption > 0:
            self.battery_level -= consumption
            if self.battery_level < 0:
                self.battery_level = 0
            self.load_curve.append(0)
        elif consumption == 0:
            potential_battery_level = self.battery_level + self.charging_power
            if potential_battery_level >= battery_capacity:
                over_charged_value = potential_battery_level - battery_capacity
                new_charging_value = self.charging_power - over_charged_value
                self.battery_level += new_charging_value
                self.load_curve.append(new_charging_value)
            else:
                self.battery_level += self.charging_power
                self.load_curve.append(self.charging_power)
        else:
            print("negative consumption not possible")

        self.battery_level_curve.append(self.battery_level)
        return self.battery_level

    def calc_soc(self, battery_capacity, battery_efficiency):
        # Convert battery efficiency percentage to a decimal
        battery_efficiency_decimal = battery_efficiency / 100
        # Calculate the energy available in the battery
        available_energy = self.battery_level * battery_efficiency_decimal
        # Calculate the state of charge (SoC)
        self.soc = (available_energy / battery_capacity) * 100
        self.soc_curve.append(self.soc)
        return self.soc


class ElectricVehicleFlatCharge(ElectricVehicle):
    def __init__(self, name, unique_id, **params):
        super().__init__(name, unique_id)
        self.max_power = 3.7
        self.min_power = 0

    def print(self):
        print("hello")




# self.min_soc xxxx
# self.range_anxiety


if __name__ == '__main__':
    # initialize car object and retrieve expected battery capacity value
    bmw_i3 = ElectricVehicle("bmw_i3", 1)
    small_capacity = bmw_i3.small_battery_capacity
    charging_power = bmw_i3.charging_power
    for i in [10, 20, 10, 0]:
        battery_level = bmw_i3.calculate_battery_level(consumption=i, battery_capacity=small_capacity)
        soc = bmw_i3.calc_soc(small_capacity, 100)
    # print(bmw_i3.battery_level_curve)
    # print(bmw_i3.load_curve)
    # print(bmw_i3.soc_curve)

    next_car = ElectricVehicleFlatCharge("bmw_i3", 1)
    next_small = next_car.small_battery_capacity
    for i in [10, 20, 10, 0]:
        battery_level1 = next_car.calculate_battery_level(consumption=i, battery_capacity=next_small)
        soc1 = next_car.calc_soc(small_capacity, 100)
    print(next_car.battery_level_curve)

# TODO build new class with results, access results easier and write functions to aggregate them
