import math
import pandas as pd

class ElectricityGrid:
    def __init__(self, number_of_cars: int, cars_per_bus: int, charging_power: str):
        self.number_of_cars = number_of_cars
        self.cars_per_bus = cars_per_bus
        self.num_of_busses = None

        self.bus_kw_limit = 4000
        self.buses = {}

        # self.households_germany = 41500000
        # self.number_busses = 8000
        self.charging_power = charging_power
        self._set_charging_power()
        self._create_buses()

    def _set_charging_power(self):
        charging_dict = {
            'low': 3.7,
            'normal': 11,
            'high': 22
        }

        if self.charging_power == 'low':
            self.charging_power_home = charging_dict['low']
        elif self.charging_power == 'normal':
            self.charging_power_home = charging_dict['normal']
        elif self.charging_power == 'high':
            self.charging_power_home = charging_dict['high']
        else:
            raise Exception("Please enter charging scenario low, normal or high.")

    def _create_buses(self):
        self.num_of_busses = math.ceil(self.number_of_cars / self.cars_per_bus)

    def create_cars(self):
        for car in range(self.number_of_cars):
            print(car)

    def adjust_charging_power(self):
        pass

    def set_up_grid(self):
        # grid.create_cars()
        grid._create_buses()

    def assign_cars_to_buses(self):
        # initialize variables
        car_id = 0

        # loop through each bus
        for bus_id in range(self.num_of_busses):
            df = pd.DataFrame(columns=['car_id'])
            for i in range(self.cars_per_bus):
                # add the car to the current dataframe
                if car_id < self.number_of_cars:
                    df.loc[i] = [car_id]
                    car_id += 1

            df['data'] = [pd.DataFrame(columns=['timestamp', 'battery_level', 'load_curve', 'soc'])] * len(df)

            # add the dataframe to the bus_dict with a key equal to the bus ID
            self.buses[bus_id] = df

    def get_mobility_data(self, bus_id):
        data = self.buses[bus_id]['data'][0]
        print(data)
        return data


grid = ElectricityGrid(909, 100, 'low')
grid.set_up_grid()
grid.assign_cars_to_buses()

data = grid.get_mobility_data(2)