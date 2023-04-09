# Initialization of the model
# Retrieve the EV Agent data
# create electric vehicle with battery
# assign battery capacity
# initialize electric vehicle
import json
import pandas as pd


class ElectricVehicle:
    def __init__(self, car_model, unique_id, battery_size):
        self.car_model = car_model
        self.unique_id = unique_id
        self.battery_size = battery_size

        self.battery_level = None
        self.battery_capacity = None
        self.soc = None

        self.__initialize_car_values()

        self.load_curve = []
        self.battery_level_curve = []
        self.soc_curve = []

    def __initialize_car_values(self):
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        if self.battery_size == 'small':
            self.battery_capacity = car_dict[self.car_model]["small_battery"]
        elif self.battery_size == 'normal':
            self.battery_capacity = car_dict[self.car_model]["normal_battery"]
        elif self. battery_size == 'large':
            self.battery_capacity = car_dict[self.car_model]["large_battery"]
        else:
            print("Define input parameter for battery size: 'small', 'normal' or 'large'")
        self.number_of_car = car_dict[self.car_model]["number"]
        self.charging_power = car_dict[self.car_model]["charging_power"]

    def calculate_battery_level(self, consumption, battery_efficiency):
        if self.battery_level is None:
            self.battery_level = self.battery_capacity

        if consumption > 0:
            self.battery_level -= consumption
            if self.battery_level < 0:
                self.battery_level = 0
            self.load_curve.append(0)
        elif consumption == 0:
            potential_battery_level = self.battery_level + self.charging_power
            if potential_battery_level >= self.battery_capacity:
                over_charged_value = potential_battery_level - self.battery_capacity
                new_charging_value = self.charging_power - over_charged_value
                self.battery_level += new_charging_value
                self.load_curve.append(new_charging_value)
            else:
                self.battery_level += self.charging_power
                self.load_curve.append(self.charging_power)
        else:
            print("negative consumption not possible")

        self.battery_level_curve.append(self.battery_level)
        self.__calc_soc(battery_efficiency)
        return self.battery_level

    def __calc_soc(self, battery_efficiency):
        # Convert battery efficiency percentage to a decimal
        battery_efficiency_decimal = battery_efficiency / 100
        # Calculate the energy available in the battery
        available_energy = self.battery_level * battery_efficiency_decimal
        # Calculate the state of charge (SoC)
        self.soc = (available_energy / self.battery_capacity) * 100
        self.soc_curve.append(self.soc)


class ElectricVehicleFlatCharge(ElectricVehicle):
    def __init__(self, car_model, unique_id, battery_size, **params):
        super().__init__(car_model, unique_id, battery_size)
        self.max_power = 3.7
        self.min_power = 0

    #TODO
    # Implement flat charging calculation
    # implement soc to stop charging
    # check each timestamp if soc already reached
    # TODO implement rance anxiety
    # self.min_soc xxxx
    # self.range_anxiety
    def print(self):
        print("hello")






if __name__ == '__main__':
    # initialize car object and retrieve expected battery capacity value
    bmw_i3 = ElectricVehicle("bmw_i3", 1, 'small')

    # todo use the consumption data from mobility data aggregator
    for i in [10, 20, 10, 0]:
        battery_level = bmw_i3.calculate_battery_level(consumption=i, battery_efficiency=100)

    # print(bmw_i3.battery_level_curve)
    print(bmw_i3.load_curve)
    print(bmw_i3.soc_curve)

# TODO build new class with results, access results easier and write functions to aggregate them
# TODO aggregated mobility file - read it whole - store maybe as dict
# TODO have timestep as key and consumption as values