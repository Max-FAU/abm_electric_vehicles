from mesa import Agent, Model
import json
import pandas as pd
import auxiliary as aux
import random
import datetime
from mobility_data import MobilityDataAggregator
from mesa.time import SimultaneousActivation
import numpy as np


class ElectricVehicle(Agent):
    valid_models = ['bmw_i3', 'renault_zoe', 'tesla_model_3', 'vw_up', 'vw_id3', 'smart_fortwo', 'hyundai_kona', 'fiat_500', 'vw_golf', 'vw_id4_id5']
    # Track already assigned mobility profiles
    picked_mobility_data = []

    def __init__(self, unique_id, model, car_model, start_date, end_date):
        super().__init__(unique_id, model)
        # Initialize Agent attributes from input
        self.unique_id = unique_id
        assert car_model in ElectricVehicle.valid_models, f"Invalid car model: '{car_model}'. Must be one of: {', '.join(ElectricVehicle.valid_models)}"
        self.car_model = car_model
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # Initialize Agent attributes from json file car_values.json when creating an Agent
        self.car_values = dict()
        self.battery_capacity = None
        self.car_size = None
        self.number_of_car_model = None
        self.charging_power_ac = None
        self.charging_power_dc = None
        self.complete_initialization()

        self.mobility_data = None
        self.car_id = None
        self.add_mobility_data()

        # Timestamp is set in step function
        self._timestamp = None

        # All data from mobility file for the set timestamp
        self.trip_number = None
        self.deltapos = None
        self.cluster = None
        self.consumption = None
        self.panel_session = None
        self.terminal = None

        # Data from calculations for the timestamp
        self.plug_in_buffer = None
        self.plug_in_status = None
        self.battery_level = None

    # TODO REFACTOR GETTER AND SETTER IN PYTHONIC WAY
    @property
    def timestamp(self):
        """Function to get the current timestamp."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        """Function to set the timestamp of the Agent to the input timestamp."""
        self._timestamp = timestamp

    def load_car_value_dict(self):
        """Opens json file with all values stored for cars."""
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)
        self.car_values = car_dict

    def set_battery_capacity(self):
        """Set battery capacity for the car model, based on the current car model and the json file."""
        self.battery_capacity = self.car_values[self.car_model]["battery_capacity"]

    def get_car_model(self):
        """Return car_model"""
        return self.car_model

    def set_car_size(self):
        """
        Sort all cars models in the car_values dict according to their battery capacity
        and return the position of the given car model.

        Example:
            Before sorting:
            Name   | battery size |  Position
            Car A  |      10      |  Position 0
            Car B  |      20      |  Position 1
            Car C  |      15      |  Position 2

            After sorting:
            Name   | battery size |  Position
            Car A  |      10      |  Position 0
            Car C  |      15      |  Position 1
            Car B  |      20      |  Position 2

            Search now for the Name in the list, and return the position of the Name.
        """
        sorted_models = sorted(self.car_values, key=lambda x: self.car_values[x]['battery_capacity'])
        car_model = self.get_car_model()
        self.car_size = sorted_models.index(car_model)

    def get_max_car_size(self) -> int:
        """ Return the position of the car model with the largest battery_capacity """
        sorted_models = sorted(self.car_values, key=lambda x: self.car_values[x]['battery_capacity'])
        car_model_max_size = sorted_models[-1]
        return sorted_models.index(car_model_max_size)

    def set_number_of_car_model(self):
        """Set the number of the specific car model, this is needed to calculate the distribution."""
        self.number_of_car_model = self.car_values[self.car_model]["number"]

    def set_charging_power_ac(self):
        """Set the maximum charging power the car model is capable in AC charging."""
        self.charging_power_ac = self.car_values[self.car_model]["charging_power_ac"]

    def set_charging_power_dc(self):
        """Set the maximum charging power the car model is capable in DC charging."""
        self.charging_power_dc = self.car_values[self.car_model]["charging_power_dc"]

    def complete_initialization(self):
        """
        Run all functions to initialize the car values for the model.
        These functions are all based on the 'car_model'.
        """
        self.load_car_value_dict()
        self.set_battery_capacity()
        self.set_car_size()
        self.set_number_of_car_model()
        self.set_charging_power_ac()
        self.set_charging_power_dc()

    def load_matching_df(self) -> pd.DataFrame:
        """
        Load the csv file having the matching between median trip length in the mobility file with the battery size of the car.
        Small cars get matched with mobility files having, based on median trip length, shorter trips for the time period
        of the Simulation.
        """
        # this file generated with one time run in match_cars_mobility.py
        file_name_median_trip_len = 'median_trip_length.csv'
        try:
            df = pd.read_csv(file_name_median_trip_len, index_col=0)
        except:
            print("Mobility mapping file has to be generated once.\n"
                  "This might take a while.")
            self.create_matching_file(file_name_median_trip_len)
            df = pd.read_csv(file_name_median_trip_len, index_col=0)
        return df

    def create_matching_file(self, file_name_median_trip_len):
        """If the matching csv file cannot be loaded, it will be generated."""
        directory_path = aux.get_directory_path()
        no_clusters = self.get_max_car_size()
        aux.create_median_trip_length_file(directory_path=directory_path,
                                           start_date=self.start_date,
                                           end_date=self.end_date,
                                           no_deciles=no_clusters,
                                           file_name=file_name_median_trip_len)

    # TODO Refactor and split into smaller functions
    def create_potential_matches(self, df) -> pd.DataFrame:
        # check all already picked 'car_ids' and filter them out of df
        df = df.loc[~df['car_id'].isin(ElectricVehicle.picked_mobility_data)]
        # if the dataframe is empty clean the picked cars and start again / mobility data can be picked twice
        if df.empty:
            ElectricVehicle.picked_mobility_data.clear()
            df = df.loc[~df['car_id'].isin(ElectricVehicle.picked_mobility_data)]
        return df

    # TODO Refactor and split into smaller functions
    def create_final_match(self, df) -> int:
        # Get the closest number in column decile_label to the car_size of the current Agent
        closest_number = min(df['decile_label'], key=lambda x: abs(x - self.car_size))
        # Get a list of indexes where decile_label is equal to closest_number
        matching_indexes = df.index[df['decile_label'] == closest_number].tolist()
        # Get a random index from the matching indexes
        random_index = random.choice(matching_indexes)
        # find the car id
        random_car_id = df.loc[random_index, 'car_id']
        # add the car id to already picked ids
        ElectricVehicle.picked_mobility_data += [random_car_id]
        return random_car_id

    def set_car_id(self, car_id):
        self.car_id = car_id

    def load_mobility_data(self, file_path) -> pd.DataFrame:
        # Read only used columns to speed up data reading
        raw_mobility_data = pd.read_csv(file_path, usecols=['TIMESTAMP', 'TRIPNUMBER', 'DELTAPOS', 'CLUSTER', 'ECONSUMPTIONKWH', 'ID_PANELSESSION', 'ID_TERMINAL'])
        data_aggregator = MobilityDataAggregator(raw_mobility_data=raw_mobility_data,
                                                 start_date=self.start_date,
                                                 end_date=self.end_date)

        return data_aggregator.df_processed

    def add_mobility_data(self):
        """Function to assign the correct mobility file to a car and load it afterwards."""
        # Load a matching file, it will be generated and then loaded, if not existing
        df = self.load_matching_df()
        df = self.create_potential_matches(df)
        car_id = self.create_final_match(df)
        self.set_car_id(car_id=car_id)

        # Load correct mobility file
        file_path = aux.create_file_path(car_id, test=True)
        self.mobility_data = self.load_mobility_data(file_path)
        print("... mobility data for car {} loaded successfully.".format(self.car_id))

    def get_mobility_data(self):
        return self.mobility_data

    def set_data_current_timestamp(self):
        df = self.get_mobility_data()
        timestamp = self.timestamp
        self.set_trip_number(df, timestamp)
        self.set_deltapos(df, timestamp)
        self.set_cluster(df, timestamp)
        self.set_consumption(df, timestamp)
        self.set_panel_session(df, timestamp)
        self.set_terminal(df, timestamp)

    def set_trip_number(self, df, timestamp):
        self.trip_number = df.loc[timestamp, 'TRIPNUMBER']

    def set_deltapos(self, df, timestamp):
        self.deltapos = df.loc[timestamp, 'DELTAPOS']

    def set_cluster(self, df, timestamp):
        self.cluster = df.loc[timestamp, 'CLUSTER']

    def set_consumption(self, df, timestamp):
        self.consumption = df.loc[timestamp, 'ECONSUMPTIONKWH']

    def set_panel_session(self, df, timestamp):
        self.panel_session = df.loc[timestamp, 'ID_PANELSESSION']

    def set_terminal(self, df, timestamp):
        self.terminal = df.loc[timestamp, 'ID_TERMINAL']

    def get_last_trip_id(self) -> int:
        """Returns the trip number of the last trip of the day."""
        return max(self.mobility_data['TRIPNUMBER'])

    def get_current_trip_id(self) -> int:
        """Return the trip number of the current trip."""
        return self.mobility_data.loc[self.timestamp, 'TRIPNUMBER']

    # TODO Implement that there is no key error for checking for the next trip at the end of the file
    def get_next_trip_id(self) -> int:
        """Return the trip number of the next trip."""
        current_trip = self.get_current_trip_id()
        return current_trip + 1

    def get_consumption_of_trips(self) -> pd.DataFrame:
        """Returns a dataframe with consumptions for all trips."""
        return self.mobility_data.groupby('TRIPNUMBER')['ECONSUMPTIONKWH'].sum()

    def get_consumption_of_trip_id(self, trip_id) -> float:
        """Returns the consumption of a specific trip."""
        df_consumption_trips = self.get_consumption_of_trips()
        return df_consumption_trips.loc[trip_id]

    def get_consumption_with_range_anxiety(self, trip_id, anxiety_factor=1.5):
        """Returns the consumption, added the range anxiety factor to it."""
        consumption_spec_trip = self.get_consumption_of_trip_id(trip_id)
        return consumption_spec_trip * anxiety_factor

    def get_battery_capacity(self):
        """Getter function to return the total battery capacity of a car model."""
        return self.battery_capacity

    # TODO REFACTOR THE SETTER AND GETTER METHODS WTIH PROPERTIES
    def get_battery_level(self):
        """Getter function to return the current battery level."""
        return self.battery_level

    def set_battery_level(self, value):
        """Setter function to set the battery level."""
        self.battery_level = value

    def calc_new_battery_level(self):
        old_battery_lvl = self.get_battery_level()
        if old_battery_lvl is None:
            old_battery_lvl = self.battery_capacity
        new_battery_lvl = old_battery_lvl - self.consumption
        self.set_battery_level(new_battery_lvl)

    @property
    def soc(self):
        """Calculate battery level percentage remaining"""
        return round((self.battery_level / self.battery_capacity) * 100, 2)

    def step(self):
        if self.timestamp is None:
            self.timestamp = self.start_date
        else:
            # Each step add 15 minutes
            self.timestamp += datetime.timedelta(minutes=15)

        self.set_data_current_timestamp()
        self.calc_new_battery_level()
        print(self.soc)


class ChargingModel(Model):
    def __init__(self,
                 num_agents: int,
                 start_date: str,
                 end_date: str):

        self.start_date = start_date
        self.end_date = end_date
        self.num_agents = num_agents  # agents are number of EV Agents
        self.schedule = SimultaneousActivation(self)

        self.list_models = self.generate_cars_according_to_dist()

        i = 0
        while i < len(self.list_models):
            car_model = self.list_models[i]
            try:
                agent = ElectricVehicle(unique_id=i,
                                        model=self,
                                        car_model=car_model,
                                        start_date=self.start_date,
                                        end_date=self.end_date)
                self.schedule.add(agent)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("Added agent number {} to the model.".format(i))
            i += 1

    def generate_cars_according_to_dist(self):
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

        car_models = np.random.choice(cars, size=self.num_agents, p=distribution)
        # print(len(car_names), "car names generated.")

        return car_models

    def step(self):
        # step through schedule
        self.schedule.step()


if __name__ == '__main__':
    model = ChargingModel(num_agents=1,
                          start_date='2008-07-13',
                          end_date='2008-07-14')

    for i in range(96):
        model.step()