from mesa import Agent, Model
import json
import pandas as pd
import auxiliary as aux
import random
import datetime
from mobility_data import MobilityDataAggregator
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
import numpy as np


class ElectricVehicle(Agent):
    # Only the valid models below are implemented yet
    valid_models = ['bmw_i3', 'renault_zoe', 'tesla_model_3', 'vw_up', 'vw_id3', 'smart_fortwo', 'hyundai_kona', 'fiat_500', 'vw_golf', 'vw_id4_id5']
    # Track already assigned mobility profiles
    picked_mobility_data = []

    def __init__(self, unique_id, model, car_model, start_date, end_date, target_soc, max_transformer_capacity):
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

        # self.charger_to_charger_trips = self.set_charger_to_charger_trips()
        self.max_transformer_capacity = max_transformer_capacity
        self.consumption_to_next_charge = None
        self.charging_duration = None
        self.charging_priority = None

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

    # TODO Refactor and split into smaller functions or maybe create a new class
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
        # self.mobility_data.to_csv("mobility_test_data.csv")
        # self.mobility_data = self.load_mobility_data('mobility_test_data.csv')

    def set_consumption_to_next_charge(self):
        """
        Next trip is not defined by trip number but on leaving a charger and reaching a new charger.
        No public chargers considered here.
        """
        # Make a copy of the mobility data to avoid modifying the original DataFrame
        mobility_data = self.mobility_data.copy()
        df_slice = mobility_data.loc[self.timestamp:]
        aux.set_print_options()
        block_sum = 0
        for i in range(len(df_slice)):
            if df_slice.iloc[i]['CLUSTER'] == 0:
                block_sum += df_slice.iloc[i]['ECONSUMPTIONKWH']
            elif block_sum != 0:
                self.consumption_to_next_charge = block_sum
                break
        else:
            self.consumption_to_next_charge = 0

    def get_consumption_to_next_charge(self):
        return self.consumption_to_next_charge

    def set_charging_duration(self):
        mobility_data = self.get_mobility_data()

        current_timestamp = self.timestamp
        df_slice = mobility_data.loc[current_timestamp:]
        cluster_changes = df_slice[df_slice['CLUSTER'].eq(0)]

        if not cluster_changes.empty:
            next_cluster_change = cluster_changes.index[-1]
        else:
            next_cluster_change = df_slice.index[-1]
            # duration = pd.Timedelta('0 days 00:00:00')
        duration = next_cluster_change - current_timestamp
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
        # TODO in CONSUMPTION efficiency is already included, in charging_power station / car it is not included
        empty_battery_capacity = self.empty_battery_capacity()   # kwh
        possible_soc_capacity = self.empty_battery_capacity_soc()    # kwh

        # Set correct charging power for the car based on cluster
        self.set_charging_power_car()
        charging_power_car = self.get_charging_power_car()
        charging_value_car = charging_power_car / 4   # kwh

        # Set correct charging power for the station based on cluster
        self.set_charging_power_station()
        charging_power_station = self.get_charging_power_station()
        charging_value_station = charging_power_station / 4    # kwh

        possible_charging_value = min(empty_battery_capacity,
                                      possible_soc_capacity,
                                      charging_value_car,
                                      charging_value_station)

        return possible_charging_value

    def set_charging_power(self):
        charging_value = self.get_charging_value()
        self.charging_power = charging_value * 4   # to get kW

    # TODO efficiency to 100 digits
    # grid value will be the same // grid load is the wrong name for it
    def set_grid_load(self, charging_efficiency=0.95):
        charging_value = self.get_charging_value()
        charging_power = charging_value * 4   # kW
        # TODO GRID LOAD IS IMPORTANT AND IS WRONG
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

        # Both are set to ac charging because we consider work has also ac chargers
        if cluster == 1:  # home
            self.charging_power_car = ac_charging_capacity
        elif cluster == 2:  # work
            self.charging_power_car = ac_charging_capacity
        else:
            # Change this, if car should charge everywhere e.g. public charging
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

        # TODO NEXT TRIP NEEDS DOES NOT WORK
        self.set_consumption_to_next_charge()
        consumption_next_trip = self.get_consumption_to_next_charge()
        consumption_next_trip_range_anx = self.get_consumption_with_range_anxiety(consumption_next_trip)
        battery_capacity = self.get_battery_capacity()

        # Calculate the consumption next trip related to the battery capacity
        # Next trip needs a large amount of battery capacity then prioritize charging
        relative_need = round(consumption_next_trip_range_anx / battery_capacity * 100, 0)

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

        debug = False
        if debug:
            print('soc {}, prio_soc {},'
                  'next_trip {}, prio_next_trip {}, '
                  'charging_duration {}, prio_time {}'.format(self.soc,
                                                              prio_soc,
                                                              relative_need,
                                                              prio_next_trip,
                                                              charging_duration,
                                                              prio_time))

        self.charging_priority = prio_soc + prio_next_trip + prio_time

    def get_charging_priority(self):
        return self.charging_priority

    def get_unique_id(self):
        return self.unique_id

    def get_car_id(self):
        return self.car_id

    def get_max_transformer_capacity(self):
        return self.max_transformer_capacity

    def interaction_charging_values(self):
        from grid_agent import PowerCustomer
        customer = PowerCustomer(yearly_cons_household=3500, start_date=self.start_date, end_date=self.end_date)
        customer.set_current_load(self.timestamp)
        customer.set_current_load_kw()
        customer_base_load = customer.get_current_load_kw()

        transformer_capacity = self.get_max_transformer_capacity()
        max_capacity = transformer_capacity - customer_base_load * len(ElectricVehicle.picked_mobility_data)

        # This action is done in every step
        all_agents = self.model.schedule.agents

        # Filter to keep only ElectricVehicles (Filter out transformers)
        car_agents = []
        for agent in all_agents:
            if isinstance(agent, ElectricVehicle):
                car_agents.append(agent)

        from interaction import InteractionClass

        model_interaction = InteractionClass(car_agents)
        all_charging_agents = model_interaction.get_all_charging_agents()
        all_priorities = model_interaction.get_all_priorities()
        total_charging_power = model_interaction.get_total_charging_power()

        if total_charging_power > max_capacity:
            highest_priority = max(all_priorities)
            lowest_priority = min(all_priorities)
            agents_higher_priority = []

            available_capacity = max_capacity

            for priority in range(highest_priority, lowest_priority - 1, -1):
                agents_priority = model_interaction.get_agents_with_charging_priority(priority)

                sub_total_charging_power = 0

                for agent in agents_priority:
                    charging_power = agent.get_charging_value() * 4  # we need kw
                    sub_total_charging_power += charging_power

                    if sub_total_charging_power > available_capacity:
                        charging_power_per_agent = model_interaction.get_charging_power_per_agent(available_capacity, priority)
                        charging_value_per_agent = charging_power_per_agent / 4

                        charging_value_to_distribute = 0
                        agents_exceeding_charging_value = []
                        num_agents_priority = len(agents_priority)

                        for agent in agents_priority:
                            if agent.get_charging_value() < charging_value_per_agent:
                                charging_value_to_distribute += (charging_value_per_agent - agent.get_charging_value())
                                agents_exceeding_charging_value.append(agent)

                        other_agents_increase = charging_value_to_distribute / num_agents_priority

                        for agent in agents_priority:
                            agent.revert_charge()
                            if agent in agents_exceeding_charging_value:
                                charging_value = min(agent.get_charging_value(), charging_value_per_agent)
                            else:
                                charging_value = charging_value_per_agent + other_agents_increase
                            if charging_value > agent.get_charging_value():
                                charging_value = min(agent.get_charging_value(), charging_value)
                                # TODO
                                # TODO Leave away
                                # Schauen ob der neue charging value mit der "umlage" größer ist als der
                                # alte, wenn größer als der alte, dann den alten charging value nehmen
                                # differenz wieder berechnen und auf die noch nicht processed agents addieren
                                # wenn der letzte agent dass dann nicht mehr aufnehmen kann, wegfallen lassen.

                            agent.set_charging_value(value=charging_value)
                            agent.charge()

                        # check if higher priorities were already processed
                        for agent in all_charging_agents:
                            if agent not in agents_priority and agent not in agents_higher_priority:
                                agent.revert_charge()
                                agent.set_charging_value(value=0)
                                agent.charge()
                        break

                    agents_higher_priority.append(agent)
                available_capacity -= sub_total_charging_power

            # find all agents with the highest priority
            # add the charging power one after another to a sub total charging power for that priority
            # check continuously if the sub total charging power for that priority is higher than the max_capacity
            # if it is higher all cars of that priority needs to be reduced AND all other charging values
            # need to be set to 0

            # else
            # go to the next priority and start the process again
            # BUT set the charging value only of the agents to 0 that have lower priority than the current
            # priority, or check if agents are in a list, if they are not in the list, set the charging power
            # to 0

            # AFTER THIS revert the charge for all cars that are either set to 0 or have reduced charging power
            # charge again
            # if charging reduction is split to all charging cars with same priority
            # it have to be maximum capped at last charging value
            # the rest of charging value that is then used can be distributed to all others


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

        # Check if the step is done for the last agent in model
        all_agents = self.model.schedule.agents
        all_agents_ids = []
        for agent in all_agents:
            all_agents_ids += [agent.get_unique_id()]
        current_agent_id = self.get_unique_id()
        # Check if current agent id is the last id in list of ids of scheduled agents then interact
        if all_agents_ids[-1] == current_agent_id:
            self.interaction_charging_values()

        # self.charge()
        # self.set_grid_load()

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
        # self.list_models = self.generate_test_cars()
        num_agents = len(self.list_models)

        from grid_agent import Transformer
        transformer = Transformer(num_households=num_agents)
        self.max_capacity = transformer.get_max_capacity()

        i = 0
        while i < num_agents:
            car_model = self.list_models[i]
            try:
                agent = ElectricVehicle(unique_id=i,
                                        model=self,
                                        car_model=car_model,
                                        start_date=self.start_date,
                                        end_date=self.end_date,
                                        target_soc=100,
                                        max_transformer_capacity=self.max_capacity)
                self.schedule.add(agent)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("Added agent number {} to the model.".format(i))
            i += 1

        self.datacollector = DataCollector(
            model_reporters={
                "possible_capacity": lambda m: self.max_capacity,
                "total_recharge_power": self.get_total_recharge_power
            },
            agent_reporters={
                "timestamp": lambda a: a.timestamp,
                "recharge_value": lambda a: a.charging_value,
                "battery_level": lambda a: a.battery_level,
                "soc": lambda a: a.soc,
                "power": lambda a: a.charging_value * 4
            }
        )

    def get_total_recharge_power(self):
        total_charging_power = sum([agent.charging_value * 4 for agent in self.schedule.agents])
        return total_charging_power

    def generate_test_cars(self):
        file_path = 'car_models.txt'
        car_models = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                value = line.strip()
                car_models.append(value)
        return car_models

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
        np.savetxt('car_models.txt', car_models, fmt='%s', delimiter=' ')

        return car_models

    def step(self):
        # step through schedule
        self.schedule.step()
        self.datacollector.collect(self)


if __name__ == '__main__':
    start_date = '2008-07-11'
    end_date = '2008-07-18'

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model = ChargingModel(num_agents=20,
                          start_date=start_date,
                          end_date=end_date)

    for i in range(num_intervals):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    # aux.set_print_options()
    # print(agent_data)
    import matplotlib.pyplot as plt
    ax = model_data.plot()
    # ax.set_ylim(0, 35)
    plt.show()


    # input available capacity for charging
    # max capacity - customer household
