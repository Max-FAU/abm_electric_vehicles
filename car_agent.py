from mesa import Agent
import json
import pandas as pd
import auxiliary as aux
import random
import datetime
from mobility_data import MobilityDataAggregator
from interaction import InteractionClass
from customer_agent import PowerCustomer
import dask.dataframe as dd


class ElectricVehicle(Agent):
    # Only the valid models below are implemented yet
    valid_models = ['bmw_i3', 'renault_zoe', 'tesla_model_3', 'vw_up',
                    'vw_id3', 'smart_fortwo', 'hyundai_kona', 'fiat_500',
                    'vw_golf', 'vw_id4_id5']
    # Track already assigned mobility profiles
    picked_mobility_data = []

    def __init__(self, unique_id, model, car_model, start_date, end_date, target_soc):

        super().__init__(unique_id, model)
        # Initialize Agent attributes from input
        self.type = 'Car'
        self.unique_id = unique_id
        assert car_model in ElectricVehicle.valid_models, f"Invalid car model: '{car_model}'. Must be one of: " \
                                                          f"{', '.join(ElectricVehicle.valid_models)}"
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
        self.consumption_to_next_charge = None
        self.charging_duration = None
        self.charging_priority = None

        self.base_load = 0
        self.capacity_to_charge = None

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
        with open('input/car_values.json') as f:
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
        Load the csv file having the matching between median trip length in the mobility file with the battery size
        of the car. Small cars get matched with mobility files having, based on median trip length,
        shorter trips for the time period of the Simulation.
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

    @staticmethod
    def create_potential_matches(df) -> pd.DataFrame:
        # check all already picked 'car_ids' and filter them out of df
        df = df.loc[~df['car_id'].isin(ElectricVehicle.picked_mobility_data)]
        # if the dataframe is empty clean the picked cars and start again / mobility data can be picked twice
        if df.empty:
            ElectricVehicle.picked_mobility_data.clear()
            df = df.loc[~df['car_id'].isin(ElectricVehicle.picked_mobility_data)]
        return df

    def create_final_match(self, df) -> int:
        # Get the closest number in column decile_label to the car_size of the current Agent
        closest_number = min(df['decile_label'], key=lambda x: abs(x - self.car_size))
        # Get a list of indexes where decile_label is equal to closest_number
        matching_indexes = df.index[df['decile_label'] == closest_number].tolist()
        # Set seed to always chose the same file
        random.seed(948)
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
        # Specify data types of the columns to speed up reading csv files
        cols = ['TIMESTAMP',
                'TRIPNUMBER',
                'DELTAPOS',
                'CLUSTER',
                'ECONSUMPTIONKWH',
                'ID_PANELSESSION',
                'ID_TERMINAL']

        dtypes = {'TIMESTAMP': str,
                  'TRIPNUMBER': int,
                  'DELTAPOS': float,
                  'CLUSTER': int,
                  'ECONSUMPTIONKWH': float,
                  'ID_PANELSESSION': int,
                  'ID_TERMINAL': int}

        # Read only used columns to speed up data reading
        raw_mobility_data = dd.read_csv(file_path,
                                        usecols=cols,
                                        dtype=dtypes,
                                        parse_dates=['TIMESTAMP'])

        data_aggregator = MobilityDataAggregator(raw_mobility_data=raw_mobility_data,
                                                 start_date=self.start_date,
                                                 end_date=self.end_date)

        return data_aggregator.df_processed

    def add_mobility_data(self):
        """Function to assign the correct mobility file to a car and load it afterwards."""
        # Load a matching file, it will be generated and then loaded, if not existing
        df = self.load_matching_df()
        df = ElectricVehicle.create_potential_matches(df)
        car_id = self.create_final_match(df)
        self.set_car_id(car_id=car_id)

        # Load correct mobility file
        file_path = aux.create_file_path(car_id, test=False)
        self.mobility_data = self.load_mobility_data(file_path)
        print("Adding mobility profile of car {} to agent {} ...".format(self.car_id, self.unique_id))
        # self.mobility_data.to_csv("mobility_test_data.csv")
        # self.mobility_data = self.load_mobility_data('mobility_test_data.csv')

    def set_consumption_to_next_charge(self):
        """
        Next trip is not defined by trip number but on leaving a charger and reaching a new charger.
        No public chargers considered here.
        """
        # Create copy of dataframe
        mobility_data = self.mobility_data.copy()
        # slice the mobility data from current timestamp until rest of mobility data
        df_slice = mobility_data.loc[self.timestamp:]
        block_sum = 0
        for i in range(len(df_slice)):  # iterate over every entry in the slice
            if df_slice.iloc[i]['CLUSTER'] == 0:   # check if the 'CLUSTER' is 0
                # Add the 'ECONSUMPTION' to the block_sum if it is 0
                block_sum += df_slice.iloc[i]['ECONSUMPTIONKWH']
            # if the CLUSTER is not 0 check the block_sum, if it holds a value
            if df_slice.iloc[i]['CLUSTER'] != 0 and block_sum != 0:
                # set the consumption to next charge to block sum
                self.consumption_to_next_charge = block_sum
                break     # break the loop

        else:   # if none of the condition in the loop is met
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

    def get_next_trip_id(self) -> int:
        """Return the trip number of the next trip."""
        max_trip_number = self.get_last_trip_id()
        current_trip = self.get_current_trip_id()
        next_trip = current_trip + 1
        if next_trip > max_trip_number:
            next_trip = max_trip_number
        return next_trip

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
        charging_power_car = self.get_charging_power_car()
        charging_value_car = aux.convert_kw_kwh(kw=charging_power_car)

        # Set correct charging power for the station based on cluster
        charging_power_station = self.get_charging_power_station()
        charging_value_station = aux.convert_kw_kwh(kw=charging_power_station)

        possible_charging_value = min(empty_battery_capacity,
                                      possible_soc_capacity,
                                      charging_value_car,
                                      charging_value_station)

        return possible_charging_value

    def set_charging_power(self):
        charging_value = self.get_charging_value()
        self.charging_power = aux.convert_kw_kwh(kwh=charging_value)

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

    def get_base_load(self):
        return self.base_load

    def interaction_charging_values(self):
        all_agents = self.model.schedule.agents
        interaction = InteractionClass(all_agents)
        all_agents = interaction.get_all_agents()

        interaction.set_specific_agents(ElectricVehicle)
        cars = interaction.get_specific_agents()

        transformer_capacity = 7

        interaction.set_all_charging_agents()
        interaction.set_all_charging_values()

        interaction.set_all_priorities()
        all_priorities = interaction.get_all_priorities()
        print(all_priorities)

        interaction.set_total_charging_value()
        interaction.set_total_charging_power()

        total_charging_power = interaction.get_total_charging_power()

        if total_charging_power > transformer_capacity:
            print("interaction")
            highest_priority = max(all_priorities)
            lowest_priority = min(all_priorities)

            if highest_priority == lowest_priority:
                print("All same priority.")
                # Calculate a possible charging value for each agent
                # the charging value is capacity / number of charging agents

                # if the possible charging value is higher than calculated value
                # then reduce the charging

                # if equal, keep the original charging value

                # else (possible charging value is lower)
                # keep the original charging value and
                # calculate how much can be given to the other agents
                # (calculated charging value - original charging value)
                # When original charging value is kept (because
                # it is minimum), save the agent, it has final charging value
                #
                # THEN calculate again a possible charging value for each agent
                # reduce the number of charging agents by the agents
                # that kept the original charging value
                # do this until there are no more charging agents

            else:
                # Start with the highest priority to charge proceed
                for priority in range(highest_priority, lowest_priority - 1, -1):
                    interaction.set_priority(priority)
                    interaction.set_agents_with_charging_priority()
                    # Get all agents with priority
                    # Calculate a possible charging value for each agent
                    # the charging value is capacity / number of charging agents
                    # WITH THAT priority

                    # if the possible charging value is higher than calculated value
                    # then reduce the charging

                    # if equal, keep the original charging value

                    # else (possible charging value is lower)
                    # keep the original charging value and
                    # calculate how much can be given to the other agents
                    # (calculated charging value - original charging value)
                    # When original charging value is kept (because
                    # it is minimum), save the agent, it has final charging value
                    #
                    # THEN calculate again a possible charging value for each agent
                    # reduce the number of charging agents IN THAT PRIORITY
                    # by the agents that kept the original charging value
                    # do this until there are no more charging agents in that priority

                    # check if the total charging power of the current priority is above
                    # the available capacity
                    # Set lower priorities to 0 charging value
                    # Else
                    # go to next lower priority




    def interaction_charging_values1(self):
        all_agents = self.model.schedule.agents
        transformer_capacity = 7

        # Enter the interaction class to reduce interaction charging values method
        model_interaction = InteractionClass(all_agents, transformer_capacity)
        model_interaction.set_specific_agents(ElectricVehicle)

        # model_interaction.set_available_capacity()
        # max_capacity = model_interaction.get_available_capacity()
        max_capacity = transformer_capacity

        model_interaction.set_all_charging_agents()
        all_charging_agents = model_interaction.get_all_charging_agents()

        model_interaction.set_all_priorities()
        all_priorities = model_interaction.get_all_priorities()

        model_interaction.set_all_charging_values()
        model_interaction.set_total_charging_value()
        model_interaction.set_total_charging_power()
        total_charging_power = model_interaction.get_total_charging_power()

        if total_charging_power > max_capacity:
            highest_priority = max(all_priorities)
            lowest_priority = min(all_priorities)
            agents_higher_priority = []

            available_capacity = max_capacity

            # Starting with the highest priority to charge
            for priority in range(highest_priority, lowest_priority - 1, -1):
                model_interaction.set_priority(priority)
                model_interaction.set_agents_with_charging_priority()
                agents_priority = model_interaction.get_agents_with_charging_priority()

                sub_total_charging_power = 0

                for agent in agents_priority:
                    charging_value = agent.get_charging_value()
                    charging_power = aux.convert_kw_kwh(kwh=charging_value)
                    sub_total_charging_power += charging_power

                    if sub_total_charging_power > available_capacity:
                        model_interaction.set_charging_power_per_agent()
                        charging_power_per_agent = model_interaction.get_charging_power_per_agent()
                        charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)

                        charging_value_to_distribute = 0
                        agents_lower_charging_value = []
                        num_agents_priority = len(agents_priority)

                        for agent in agents_priority:
                            if agent.get_charging_value() < charging_value_per_agent:
                                charging_value_to_distribute += (charging_value_per_agent - agent.get_charging_value())
                                agents_lower_charging_value.append(agent)

                        other_agents_increase = charging_value_to_distribute / num_agents_priority

                        for agent in agents_priority:
                            agent.revert_charge()
                            if agent in agents_lower_charging_value:
                                charging_value = min(agent.get_charging_value(), charging_value_per_agent)
                            else:
                                charging_value = charging_value_per_agent + other_agents_increase
                            print(charging_value, agent.get_charging_value())
                            # TODO CHECK THIS
                            if charging_value > agent.get_charging_value():
                                charging_value = min(agent.get_charging_value(), charging_value)

                            agent.set_charging_value(value=charging_value)
                            agent.charge()

                        # check if higher priorities were already processed
                        for agent in all_charging_agents:
                            if agent not in agents_priority \
                                    and agent not in agents_higher_priority:
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

        # Set mobility data for current timestamp
        self.set_data_current_timestamp()

        self.calc_new_battery_level()
        self.set_soc()
        self.set_target_soc_reached()

        self.set_plug_in_status()
        self.set_charging_power_car()
        self.set_charging_power_station()

        self.set_all_charging_values()
        # self.calc_charging_value()
        self.charge()
        self.set_grid_load()
        self.set_car_charging_priority()

        # Check if the step is done for the last agent in model
        # Start the interaction
        # TODO set this to car agents
        all_agents = self.model.schedule.agents
        all_agents_ids = []
        for agent in all_agents:
            all_agents_ids += [agent.get_unique_id()]
        current_agent_id = self.get_unique_id()
        # Check if current agent id is the last id in list of ids of scheduled agents then interact
        if all_agents_ids[-1] == current_agent_id:
            # Calculate how much capacity is available for charging cars after household base load
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

    # input available capacity for charging
    # max capacity - customer household







if __name__ == '__main__':
    start_date = '2008-07-13'
    end_date = '2008-07-27'
    customer = PowerCustomer(3500,
                             start_date,
                             end_date)

    agent = ElectricVehicle(unique_id=1,
                            model=None,
                            car_model='bmw_i3',
                            start_date=start_date,
                            end_date=end_date,
                            target_soc=100,
                            max_transformer_capacity=20,
                            power_customer=customer)

    # TODO überprüfen
    agent.set_consumption_to_next_charge()
