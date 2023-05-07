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
    # V
    valid_models = ['bmw_i3', 'renault_zoe', 'tesla_model_3', 'vw_up', 'vw_id3', 'smart_fortwo', 'hyundai_kona', 'fiat_500', 'vw_golf', 'vw_id4_id5']
    # Track already assigned mobility profiles
    picked_mobility_data = []

    def __init__(self, unique_id, model, car_model, start_date, end_date, target_soc):
        super().__init__(unique_id, model)
        # Initialize Agent attributes from input
        self.unique_id = unique_id
        assert car_model in ElectricVehicle.valid_models, f"Invalid car model: '{car_model}'. Must be one of: {', '.join(ElectricVehicle.valid_models)}"
        self.car_model = car_model
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.target_soc = target_soc

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
        self.battery_level = None
        self.plugged_in = None
        self.plug_in_buffer = True
        self.target_soc_reached = False
        self.soc = None

        # self.current_charging_power = None
        self.charging_power_car = None
        self.charging_power_station = None
        self.charging_value = None
        self.grid_load = None

        self.charger_to_charger_trips = self.set_charger_to_charger_trips()
        self.charging_duration = None

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

    def get_charging_power_ac(self):
        return self.charging_power_ac

    def set_charging_power_dc(self):
        """Set the maximum charging power the car model is capable in DC charging."""
        self.charging_power_dc = self.car_values[self.car_model]["charging_power_dc"]

    def get_charging_power_dc(self):
        return self.charging_power_dc

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

    # TODO Refactor and split into smaller functions or maybe create a new file
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

    # TODO RENAME PROCESS MOBILITY DATA
    def set_charger_to_charger_trips(self) -> pd.DataFrame:
        """
        Next trip is not defined by trip number but on leaving a charger and reaching a new charger.
        No public chargers considered here.
        """
        mobility_data = self.mobility_data.copy()

        mobility_data['REAL_TRIP'] = 0
        counter = 0
        last_cluster = None

        # iterate over all rows in the mobility dataframe
        for i, row in mobility_data.iterrows():
            # first row is just the counter
            if i == 0:
                row['REAL_TRIP'] = counter
            # then compare with previous cluster
            else:
                if row['CLUSTER'] == last_cluster:
                    row['REAL_TRIP'] = counter
                else:
                    # only add 1 if we reach a charger at cluster 1 home, or cluster 2 work
                    if row['CLUSTER'] == 1 or row['CLUSTER'] == 2:
                        counter += 1
                    row['REAL_TRIP'] = counter
                last_cluster = row['CLUSTER']
            # set the trip number in NEXT
            mobility_data.loc[i, 'REAL_TRIP'] = row['REAL_TRIP']
        return mobility_data

    def set_charging_duration(self):
        mobility_data = self.get_mobility_data()

        current_timestamp = self.timestamp
        df_slice = mobility_data.loc[current_timestamp:]
        cluster_changes = df_slice[df_slice['CLUSTER'].isin([0])]
        if not cluster_changes.empty:
            next_cluster_change = cluster_changes.index[0]
            duration = next_cluster_change - current_timestamp
        else:
            duration = pd.Timedelta('0 days 00:00:00')
        duration_minutes = duration.total_seconds() / 60
        duration_hours = duration_minutes / 60
        self.charging_duration = duration_hours

    def get_charging_duration(self):
        return self.charging_duration

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

    def get_consumption(self):
        """Returns consumption of current timestamp."""
        return self.consumption

    def set_panel_session(self, df, timestamp):
        self.panel_session = df.loc[timestamp, 'ID_PANELSESSION']

    def set_terminal(self, df, timestamp):
        self.terminal = df.loc[timestamp, 'ID_TERMINAL']

    def get_terminal(self):
        """Returns terminal id of current timestamp."""
        return self.terminal

    def get_last_trip_id(self) -> int:
        """Returns the trip number of the last trip of the day."""
        return max(self.mobility_data['TRIPNUMBER'])

    # TODO NEXT TRIP NEEDS SHOULD BE BASED ON -> Home to Home | Home to Work | Work to Home | Work to Work ?
    # Idea: Copy the mobility_data create a new column that checks
    # if the cluster value changed, if so add +1 to the counter in the new column, if it is the same value, do nothing
    # now we have a column counting how often it changed from 1 to 2, from 2 to 1, from 1 to 1 (if 0 between), from
    # 2 to 2 (if 0 between), this creates "real" trip ids.
    # after creating real trip ids, we can groupby real trip id and calculate the sum.

    def get_charger_to_charger_trips(self):
        return self.charger_to_charger_trips

    def get_next_trip_needs(self) -> int:
        """
        Function that only considers consumption from charger to charger, in this case
        from home to work,
        from work to home,
        from home to home,
        from work to work.
        """
        df = self.get_charger_to_charger_trips()
        current_trip_id = df.loc[self.timestamp, 'REAL_TRIP']
        next_trip_id = current_trip_id + 1

        # group by trip number to find next trip needs
        consumption_of_trips = df.groupby('REAL_TRIP')['ECONSUMPTIONKWH'].sum()
        try:
            consumption_of_next_trip = consumption_of_trips.loc[next_trip_id]
        except:
            # cannot get the trip id + 1 if it is the last trip
            consumption_of_next_trip = consumption_of_trips.loc[current_trip_id]

        return consumption_of_next_trip

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

    def get_consumption_with_range_anxiety(self, consumption_trip, anxiety_factor=1.5):
        """Returns the consumption, added the range anxiety factor to it."""
        return consumption_trip * anxiety_factor

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

    # Call this
    def calc_new_battery_level(self):
        old_battery_lvl = self.get_battery_level()
        if old_battery_lvl is None:
            old_battery_lvl = self.battery_capacity
        new_battery_lvl = old_battery_lvl - self.consumption
        self.set_battery_level(new_battery_lvl)

    def set_plug_in_buffer(self, value: bool):
        """Set plug_in buffer."""
        self.plug_in_buffer = value

    def get_plug_in_buffer(self):
        """Return current plug_in buffer."""
        return self.plug_in_buffer

    def set_soc(self):
        """Calculate battery level percentage remaining"""
        battery_level = self.get_battery_level()
        battery_capacity = self.get_battery_capacity()
        self.soc = (battery_level / battery_capacity) * 100

    def get_soc(self):
        return self.soc

    def get_target_soc(self):
        return self.target_soc

    def set_plugged_in(self, value: bool):
        self.plugged_in = value

    def get_plugged_in(self):
        return self.plugged_in

    # Call this
    def set_plug_in_status(self):
        """Function to check the plug in buffer and consumption to set the right plug in status."""
        consumption = self.get_consumption()
        plug_in_buffer = self.get_plug_in_buffer()

        if consumption == 0:
            if plug_in_buffer is True:
                self.set_plug_in_buffer(False)
                self.set_plugged_in(False)
            else:
                self.set_plugged_in(True)
        else:
            if plug_in_buffer is True:
                self.set_plugged_in(False)
            else:
                self.set_plugged_in(False)
                self.set_plug_in_buffer(True)

    def set_target_soc_reached(self):
        soc = self.get_soc()
        target_soc = self.get_target_soc()

        if soc >= target_soc:
            self.target_soc_reached = True
        else:
            self.target_soc_reached = False

    def get_target_soc_reached(self):
        return self.target_soc_reached

    def set_charging_value(self, value: float):
        self.charging_value = value

    # Call this
    def set_all_charging_values(self):
        plugged_in = self.get_plugged_in()
        target_soc_reached = self.get_target_soc_reached()

        if plugged_in is True and target_soc_reached is False:
            value = self.calc_charging_value()
            self.set_charging_value(value)
        if plugged_in is False:
            self.set_charging_value(0)
        if plugged_in is True and target_soc_reached is True:
            self.set_charging_value(0)

    def empty_battery_capacity(self):
        """ Calculate the capacity that is empty / not charged in a battery. """
        battery_capacity = self.get_battery_capacity()
        battery_level = self.get_battery_level()
        return battery_capacity - battery_level

    def empty_battery_capacity_soc(self):
        """ Calculate the capacity that is empty / not charged in a battery based on the soc and target soc. """
        target_soc = self.get_target_soc()
        current_soc = self.get_soc()
        battery_capacity = self.get_battery_capacity()
        potential_soc = target_soc - current_soc
        possible_charging_value = max(0, battery_capacity * potential_soc / 100)
        return possible_charging_value

    def calc_charging_value(self):
        """
        Function to calculate the real charging value.

        This charging value is in kW and considers:
        - charging power of the car
        - charging power of the charging station
        - empty battery capacity regarding total capacity
        - empty battery capacity regarding soc

        """

        empty_battery_capacity = self.empty_battery_capacity()   # kwh
        possible_soc_capacity = self.empty_battery_capacity_soc()    # kwh

        # Set correct charging power, maybe implement on different step
        self.set_charging_power_car()
        charging_power_car = self.get_charging_power_car()
        charging_value_car = charging_power_car / 4   # kwh

        self.set_charging_power_station()
        charging_power_station = self.get_charging_power_station()
        charging_value_station = charging_power_station / 4    # kwh

        possible_charging_value = min(empty_battery_capacity,
                                      possible_soc_capacity,
                                      charging_value_car,
                                      charging_value_station)

        return possible_charging_value

    # TODO efficiency to 100 digits
    # grid value will be the same // grid load is the wrong name for it
    def set_grid_load(self, charging_efficiency=0.95):
        charging_value = self.get_charging_value()
        charging_power = charging_value * 4   # kW
        self.grid_load = charging_value / charging_efficiency

    def get_cluster(self):
        """
        Function to return the location of the car.
        1 = Home
        2 = Work
        0 = Everywhere else / Public
        """
        return self.cluster

    def set_charging_power_car(self):
        """Can only charge at home or work."""
        cluster = self.get_cluster()
        ac_charging_capacity = self.get_charging_power_ac()
        dc_charging_capacity = self.get_charging_power_dc()
        # TODO maybe set work to dc_charging_capacity
        if cluster == 1:  # home
            self.charging_power_car = ac_charging_capacity
        elif cluster == 2:  # work
            self.charging_power_car = ac_charging_capacity
        else:
            # TODO change this, if car should charge everywhere
            self.charging_power_car = 0

    def get_charging_power_car(self):
        return self.charging_power_car

    def set_charging_power_station(self):
        cluster = self.get_cluster()
        home = 11
        work = 22
        public = 55  # or 22
        if cluster == 1:
            self.charging_power_station = home
        elif cluster == 2:
            self.charging_power_station = work
        else:
            self.charging_power_station = public

    def get_charging_power_station(self):
        return self.charging_power_station

    def get_charging_value(self):
        return self.charging_value

    def charge(self):
        charging_value = self.get_charging_value()
        self.battery_level += charging_value

    # TODO ADD THIS TO CLASS AS ATTRIBUTE
    def set_adjusted_charging_value(self, value):
        """
        After interaction some cars have new charging values.
        To revert the previous charging, a new variable will be introduced to store the new charging values separately.
        """
        self.adjusted_charging_value = value

    def revert_charge(self):
        """ Function to revert the added charging value."""
        charging_value = self.get_charging_value()
        self.battery_level -= charging_value

    def set_car_charging_priority(self):
        """
        This priority algorithm is based on different factors, such as
        SOC, Time the EV is plugged in, Next trip consumption
        https://www.researchgate.net/publication/332142057_Priority_Determination_of_Charging_Electric_Vehicles_based_on_Trip_Distance
        """
        soc = self.get_soc()
        if soc <= 20:
            prio_soc = 3
        elif 20 < soc < 80:
            prio_soc = 2
        else:
            prio_soc = 1

        consumption_next_trip = self.get_next_trip_needs()
        consumption_next_trip_range_anx = self.get_consumption_with_range_anxiety(consumption_next_trip)
        battery_capacity = self.get_battery_capacity()

        # Calculate the consumption next trip related to the battery capacity
        # Next trip needs a large amount of battery capacity then prioritize charging
        relative_need = consumption_next_trip_range_anx / battery_capacity * 100

        if relative_need <= 20:
            prio_next_trip = 1
        elif 20 < relative_need < 80:
            prio_next_trip = 2
        else:
            prio_next_trip = 3

        # in hours
        self.set_charging_duration()
        charging_duration = self.get_charging_duration()
        if charging_duration <= 3:
            prio_time = 3
        elif 3 < charging_duration < 6:
            prio_time = 2
        else:
            prio_time = 1
        print('soc {}, prio_soc {},'
              'next_trip {}, prio_next_trip {}, '
              'charging_duration {}, prio_time {}'.format(self.soc, prio_soc, relative_need, prio_next_trip, charging_duration, prio_time))
        # TODO add this to class
        charging_priority = prio_soc + prio_next_trip + prio_time

    def get_charging_priority(self):
        return self.charging_priority

    def interaction_charging_values(self):
        # This action is done in every step
        processed_agents = 0
        all_agents = self.model.schedule.agents
        num_agents = len(all_agents)

        # TODO set the max capacity to something calculated beforehand based on grid_agent
        max_capacity = 25

        for agent in self.model.schedule.agents:
            processed_agents += 1
            if processed_agents == num_agents:
                print(num_agents)
                print("processed agents is: ", processed_agents)
                # filter all agents by electricvehicles
                car_agents = [agent for agent in all_agents if isinstance(agent, ElectricVehicle)]
                # get all charging values of all agents
                car_agents_charging_values = [agent.charging_value for agent in car_agents]
                # get only cars that are charging
                num_charging_cars = len([value for value in car_agents_charging_values if value is not None and value != 0])
                # calculate the total charging values of all car agents in the model
                car_agents_charging_values_total = sum([x for x in car_agents_charging_values if x is not None])
                # print(car_agents_charging_values_total)
                if car_agents_charging_values_total > max_capacity:
                    # TODO look for priorities for cars, and only charge cars with high priorities as long as we have sufficient power available
                    exceeding_charging_value = car_agents_charging_values_total - max_capacity  # calc the exceeding energy
                    reduction_per_agent = round(exceeding_charging_value / num_charging_cars, 1)  # calc the reduction per charging agent and round the result

                    for agent in car_agents:  # update in all car agents the charging value
                        if agent.charging_value is not None and agent.charging_value != 0:
                            # TODO This does not work because we cannot reduce everything the same (sometimes we even reduce all charging value, and end up with reducing too less)
                            agent.charging_value = max(0, agent.charging_value - reduction_per_agent)
                            # TODO after setting the new charging_value, the old charging value needs to be subtracted again from the battery
                            # TODO and the new charging value needs to be set, and the self.charge() function again
                            # TODO and the grid load needs to be reverted, and then calculated with the new charging_value




        # for agent in all_agents:
        #     # if last instance of all agents has been found
        #     # get total charging_value of all agents
        #     # reduce charging for all agents
        #     pass
        #     print(agent.charging_value)
        # car_agents = [agent for agent in all_agents if isinstance(agent, ElectricVehicle)]
        # last_agent = vars(last_agent)
        # car_agents = [agent for agent in last_agent if isinstance(agent, ElectricVehicle)]
        # print(last_agent)

    def step(self):
        if self.timestamp is None:
            self.timestamp = self.start_date
        else:
            # Each step add 15 minutes
            self.timestamp += datetime.timedelta(minutes=15)

        self.set_data_current_timestamp()

        self.calc_new_battery_level()
        self.set_soc()
        self.set_target_soc_reached()

        self.set_plug_in_status()
        self.set_all_charging_values()
        self.calc_charging_value()
        self.charge()
        self.set_grid_load()

        self.set_car_charging_priority()

        # print("charging_power_station: {}, "
        #       "charging_power_car: {}, "
        #       "soc {}, "
        #       "battery {}, "
        #       "final {},"
        #       "battery lvl {}".format(self.charging_power_station,
        #                               self.charging_power_car,
        #                               self.empty_battery_capacity_soc(),
        #                               self.empty_battery_capacity(),
        #                               self.get_charging_value(),
        #                               self.get_battery_level()))

        # self.interaction_charging_values()

        # charging_power car
        # current charging power
        # plug in status
        # pos

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
                                        end_date=self.end_date,
                                        target_soc=100)
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