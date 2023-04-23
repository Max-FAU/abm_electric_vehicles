import json
import pandas as pd
import helper
import datetime
from mobility_data import MobilityDataAggregator
import mesa
import random


class ElectricVehicle(mesa.Agent):
    # list of all picked mobility data to assign them only once
    picked_mobility_data = []

    def __init__(self, unique_id: int,
                 car_model: str,
                 target_soc: float,
                 start_date: str,
                 end_date: str,
                 model):
        """
        :param car_model: 'bmw_i3' | 'renault_zoe' | 'tesla_model_3' | 'vw_up' | 'vw_id3' | 'smart_fortwo' | 'hyundai_kona' | 'fiat_500' | 'vw_golf' | 'vw_id4_id5'
        :param target_soc: 0.00 - 1.00, charging happens until target SOC has been reached
        """
        # TODO sort them differently
        # insert the super class
        super().__init__(unique_id, model)

        self.charging_value = None
        self.charging_power_word = None
        self.charging_power_home = None
        self.number_of_car = None
        self.car_id = None
        self.unique_id = unique_id
        self.car_model = car_model
        self.car_size = None

        self.battery_capacity = None
        self.battery_level = None

        assert isinstance(target_soc, float), "Target SOC must be a float."
        self.target_soc = target_soc
        self.soc = None

        self.consumption = None

        self.anxiety_factor = 1.5
        self.range_anxiety = None

        self.plugged_in = None
        self.current_charging = None

        # run this always when creating a car agent
        self.initialize_car_values()

        self.mobility_data = None
        self.start_date = start_date
        self.end_date = end_date
        self.load_mobility_data()

        self.current_timestamp = None

    def set_timestamp(self, timestamp):
        self.current_timestamp = timestamp

    def initialize_car_values(self):
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        self.battery_capacity = car_dict[self.car_model]["battery_capacity"]
        self.number_of_car = car_dict[self.car_model]["number"]
        self.charging_power_home = car_dict[self.car_model]["charging_power_home"]
        self.charging_power_word = car_dict[self.car_model]["charging_power_work"]

        if self.car_size is None:
            sorted_models = sorted(car_dict, key=lambda x: car_dict[x]['battery_capacity'])
            self.car_size = sorted_models.index(self.car_model)

    def load_mobility_data(self):
        df = pd.read_csv('median_trip_length.csv', index_col=0)

        # TODO if there are car size == trip length otherwise pick other trip lengths
        # TODO check if num of agents > 702
        min_car_size = 0
        max_car_size = 9

        potential_car_ids_index = df.index[df['decile_label'] == self.car_size].tolist()
        # if potential_car_ids_index is empty, just create new
        potential_car_ids_index = [x for x in potential_car_ids_index if x not in ElectricVehicle.picked_mobility_data]

        random_car_index = random.choice(potential_car_ids_index)
        random_car_id = df.loc[random_car_index, 'car_id']
        ElectricVehicle.picked_mobility_data += [random_car_index]
        self.car_id = random_car_id

        # TEST True = Set local directory for mobility data
        # Load correct mobility file
        file_path = helper.create_file_path(random_car_id, test=False)

        raw_mobility_data = pd.read_csv(file_path)
        data_aggregator = MobilityDataAggregator(raw_mobility_data=raw_mobility_data,
                                                 start_date=self.start_date,
                                                 end_date=self.end_date)

        self.mobility_data = data_aggregator.df_processed
        print("... mobility data for car {} loaded successfully.".format(self.car_id))

    def get_mobility_data(self):
        return self.mobility_data

    def get_car_id(self):
        return self.car_id

    def set_plug_in_status(self):
        #TODO Replace self.get_mobility_data() with df and input is df
        panel_session = self.get_mobility_data().loc[self.current_timestamp, 'ID_PANELSESSION']
        current_index = self.get_mobility_data().index.get_loc(self.current_timestamp)

        # first step for 15 minute buffer, since 1 timestep is 15 mins away
        if self.plugged_in is None:  # first time
            previous_index = current_index
        else:
            previous_index = current_index - 1

        previous_timestamp = self.get_mobility_data().index[previous_index]
        previous_panel_session = self.get_mobility_data().loc[previous_timestamp, 'ID_PANELSESSION']

        # This implements 15 minutes buffer
        if (panel_session == 0) & (previous_panel_session == 0):
            self.plugged_in = True
        else:
            self.plugged_in = False

    def charging_possible(self, soc, target_soc, plugged_in, consumption):
        if soc > target_soc:
            return False
        if not plugged_in:
            return False
        if plugged_in and soc < target_soc and consumption == 0:
            return True

    # TODO Implement charging efficiency
    def calculate_battery_level(self, charging_efficiency=0.95):

        if self.battery_level is None:
            self.battery_level = self.battery_capacity

        self.calc_soc()

        self.consumption = self.mobility_data.loc[self.current_timestamp, 'ECONSUMPTIONKWH']

        charging = self.charging_possible(self.soc,
                                          self.target_soc,
                                          self.plugged_in,
                                          self.consumption)

        if not charging:
            potential_battery_level = self.battery_level - self.consumption
            if potential_battery_level < 0:
                new_consumption_value = min(self.battery_level, self.consumption - self.battery_level)
                self.battery_level -= new_consumption_value
            else:
                self.battery_level -= self.consumption

            self.charging_value = 0
        else: # charging
            # TODO charging power home should be charging_power -> min(charging_power_home, charging_power_station)
            potential_battery_level = self.battery_level + self.charging_power_home
            if potential_battery_level >= self.battery_capacity:
                over_charged_value = potential_battery_level - self.battery_capacity
                self.charging_value = max(0, self.charging_power_home - over_charged_value)
                self.battery_level += self.charging_value
                # self.load_curve.append(new_charging_value)
            else:
                # check for target soc and reduce charging power according to it
                self.charging_value = self.target_soc * self.battery_capacity - self.battery_level
                self.charging_value = max(self.charging_value, self.charging_power_home)
                self.battery_level += self.charging_value

    def calc_soc(self):
        # Calculate the state of charge (SoC)
        self.soc = self.battery_level / self.battery_capacity

    def next_trip_needs(self):
        """ anxiety factor 1.5 """
        last_trip = max(self.mobility_data['TRIPNUMBER'])
        next_trip = self.mobility_data.loc[self.current_timestamp, 'TRIPNUMBER'] + 1
        consumption_trips = self.mobility_data.groupby('TRIPNUMBER')['ECONSUMPTIONKWH'].sum()
        if next_trip <= last_trip:
            consumption_next_trip = consumption_trips.loc[next_trip]
        else:
            consumption_next_trip = consumption_trips.loc[next_trip - 1]

        self.range_anxiety = consumption_next_trip * self.anxiety_factor

    def step(self):
        if self.current_timestamp is None:
            self.current_timestamp = self.start_date
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
        else:
            # each step add 15 minutes
            self.current_timestamp = self.current_timestamp + datetime.timedelta(minutes=15)

        self.set_plug_in_status()
        self.calculate_battery_level()
        # print("")
        # print("my car id is: ", self.car_id)
        # print("my car model is: ", self.model)
        # print("current timestamp: ", self.current_timestamp)
        # print("my battery capacity: ", self.battery_capacity)
        # print("my battery lvl: ", self.battery_level)
        # print("my consumption: ", self.consumption)
        # print("my charging value: ", self.charging_value)
        # print("my soc is: ", self.soc)
        # print("")


# class ElectricVehicleFlatCharge(ElectricVehicle):
#     def __init__(self, model, **params):
#         super().__init__(model)
#         self.max_power = 3.7
#         self.min_power = 1.22

# TODO implement power_grid_class
# TODO implement charging only between 18:00 and 06:00
# TODO implement charging in flat manner, means calculate the time the car stands
# TODO divide the charging power by the time the car has to charge
