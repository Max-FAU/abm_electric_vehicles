# Initialization of the model
# Retrieve the EV Agent data
# create electric vehicle with battery
# assign battery capacity
# initialize electric vehicle
import json


class ElectricVehicle:
    def __init__(self, name):
        self.name = name
        self.small_battery_capacity = None
        self.normal_battery_capacity = None
        self.large_battery_capacity = None
        self._initialize_car_values()

    def _initialize_car_values(self):
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        self.small_battery_capacity = car_dict[self.name]["small"]
        self.normal_battery_capacity = car_dict[self.name]["normal"]
        self.large_battery_capacity = car_dict[self.name]["large"]
        self.number_of_car = car_dict[self.name]["number"]

    def get_battery_capacity(self, capacity_type='normal'):
        value_key = {
            "small": self.small_battery_capacity,
            "normal": self.normal_battery_capacity,
            "large": self.large_battery_capacity
        }

        return value_key[capacity_type]


if __name__ == '__main__':
    # initialize car object and retrieve expected battery capacity value
    bmw_i3 = ElectricVehicle("bmw_i3")
    small_capacity = bmw_i3.get_battery_capacity('small')
    normal_capacity = bmw_i3.get_battery_capacity('normal')
    large_capacity = bmw_i3.get_battery_capacity('large')
    number_of_car = bmw_i3.number_of_car
    print(number_of_car)
