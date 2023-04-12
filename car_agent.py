# Initialization of the model
# Retrieve the EV Agent data
# create electric vehicle with battery
# assign battery capacity
# initialize electric vehicle
import json
import pandas as pd
from mobility_data import MobilityDataAggregator


class ElectricVehicle:
    def __init__(self, model, battery_size):
        """Battery Size: small, normal, large"""
        self.model = model
        self.battery_size = battery_size

        self.unique_id = None
        self.mobility_data = None
        self.battery_level = None
        self.battery_capacity = None
        self.soc = None
        self.range_anxiety = None

        self._initialize_car_values()

        self.load_curve = []
        self.battery_level_curve = []
        self.soc_curve = []

    def _initialize_car_values(self):
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        if self.battery_size == 'small':
            self.battery_capacity = car_dict[self.model]["small_battery"]
        elif self.battery_size == 'normal':
            self.battery_capacity = car_dict[self.model]["normal_battery"]
        elif self.battery_size == 'large':
            self.battery_capacity = car_dict[self.model]["large_battery"]
        else:
            print("Define input parameter for battery size: 'small', 'normal' or 'large'")
        self.number_of_car = car_dict[self.model]["number"]
        self.charging_power = car_dict[self.model]["charging_power"]

    def calculate_battery_level(self, consumption, battery_efficiency):
        if self.battery_level is None:
            self.battery_level = self.battery_capacity

        if consumption > 0:
            self.battery_level -= consumption
            if self.battery_level < 0:
                self.battery_level = 0
            # Append 0 to load curve if there is consumption, no charging happens
            self.load_curve.append(0)
        elif consumption == 0:
            # Calculate the potential battery level after full charging
            # to compare with maximum battery capacity
            potential_battery_level = self.battery_level + self.charging_power
            if potential_battery_level >= self.battery_capacity:
                over_charged_value = potential_battery_level - self.battery_capacity
                # Reduce charging power if not enough capacity in battery left
                # cannot be negative
                new_charging_value = max(0, self.charging_power - over_charged_value)
                self.battery_level += new_charging_value
                self.load_curve.append(new_charging_value)
            else:
                self.battery_level += self.charging_power
                self.load_curve.append(self.charging_power)
        else:
            print("negative consumption not possible")

        self.battery_level_curve.append(self.battery_level)
        self._calc_soc(battery_efficiency)
        return self.battery_level

    def _calc_soc(self, battery_efficiency):
        # Convert battery efficiency percentage to a decimal
        battery_efficiency_decimal = battery_efficiency / 100
        # Calculate the energy available in the battery
        available_energy = self.battery_level * battery_efficiency_decimal
        # Calculate the state of charge (SoC)
        self.soc = (available_energy / self.battery_capacity) * 100
        self.soc_curve.append(self.soc)

    def add_mobility_data(self, mobility_data: pd.DataFrame, starting_date: str, num_days: int):
        try:
            data = MobilityDataAggregator(mobility_data)
            self.mobility_data = data.prepare_mobility_data(starting_date, num_days)
        except:
            print('Adding mobility data failed.')
        return self.mobility_data

    def calculate_range_anxiety(self):
        current_trip = ""
        next_trip = ""


class ElectricVehicleFlatCharge(ElectricVehicle):
    def __init__(self, model, battery_size, **params):
        super().__init__(model, battery_size)
        self.max_power = 3.7
        self.min_power = 0

    # TODO
    # Implement flat charging calculation
    # implement soc to stop charging
    # check each timestamp if soc already reached
    # TODO implement rance anxiety
    # self.min_soc xxxx
    # self.range_anxiety
    def print(self):
        print("hello")


if __name__ == '__main__':
    path = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    raw_mobility_data = pd.read_csv(path)
    unique_id = raw_mobility_data['ID_TERMINAL'].unique()[0]
    print(unique_id)

    # TODO Calculate average trip size and assign larger batteries for average long trips
    # initialize car object and retrieve expected battery capacity value
    bmw_i3 = ElectricVehicle(model='bmw_i3', battery_size='small')

    bmw_i3.add_mobility_data(mobility_data=raw_mobility_data,
                             starting_date='2008-07-13',
                             num_days=1)
    breakpoint()
    timestamps = []
    for timestamp, data_row in bmw_i3.mobility_data.iterrows():
        battery_level = bmw_i3.calculate_battery_level(consumption=data_row['ECONSUMPTIONKWH'],
                                                       battery_efficiency=100)
        timestamps.append(timestamp)

    # print(bmw_i3.unique_id)
    # print(timestamps)
    # print(bmw_i3.battery_level_curve)
    # print(bmw_i3.load_curve)
    # print(bmw_i3.soc_curve)

    results = pd.DataFrame(
        {
            'timestamp': timestamps,
            'battery_level': bmw_i3.battery_level_curve,
            'load_curve': bmw_i3.load_curve,
            'soc': bmw_i3.soc_curve,
            'id': bmw_i3.unique_id
        }
    ).set_index('timestamp')

    print(results)

# TODO build new class with results, access results easier and write functions to aggregate them
# TODO aggregated mobility file - read it whole - store maybe as dict
# TODO have timestep as key and consumption as values

# TODO calculate how much energy the next trip needs
# TODO implement the check for SOC, charge only until SOC has been reached
# TODO implement charging only between 18:00 and 06:00
# TODO implement charging in flat manner, means calculate the time the car stands
# TODO divide the charging power by the time the car has to charge
