from mesa import Agent
import json
import numpy as np
import pandas as pd
import auxiliary as aux
import random
import datetime
from mobility_data import MobilityDataAggregator
from agents.customer_agent import PowerCustomer
from agents.transformer_agent import Transformer
from project_paths import CAR_VALUES_PATH, MEDIAN_TRIP_LEN_PATH, CHARGER_VALUE_PATH


class ElectricVehicle(Agent):
    # Only the valid models below are implemented yet
    valid_models = ['bmw_i3', 'renault_zoe', 'tesla_model_3', 'vw_up',
                    'vw_id3', 'smart_fortwo', 'hyundai_kona', 'fiat_500',
                    'vw_golf', 'vw_id4_id5']
    # Track already assigned mobility profiles
    picked_mobility_data = []

    def __init__(self,
                 unique_id,
                 model,
                 car_model,
                 start_date: str,
                 end_date: str,
                 charging_eff: int,
                 target_soc: int,
                 charging_algo: bool,
                 seed_value: int,
                 defect: bool,
                 # TODO recency_bias:float - add this as a model level parameter,not agent level
                 ):

        super().__init__(unique_id, model)
        # Initialize Agent attributes from input
        self.type = 'Car'
        self.unique_id = unique_id
        assert car_model in ElectricVehicle.valid_models, f"Invalid car model: '{car_model}'. Must be one of: " \
                                                          f"{', '.join(ElectricVehicle.valid_models)}"
        self.car_model = car_model
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.charging_eff = charging_eff
        self.target_soc = target_soc
        self.charging_algo = charging_algo
        self.seed_value = seed_value

        # ----------------------------------------------
        # TODO work in progress section
        # Initialize Agent attributes useful for evaluation of cooperation/defection in the DR program
        self.defect = defect  # 0 or 1 - the decision to defect or not, based on the probability to defect prob_defect
        self.prob_defect = None  # [0,1] - a probability that the agent will defect
        self.wt_peer = None  # [0,1] - represents the degree of pro-social nature
        # for now, randomly set this weight in the complete_dr_initialization function
        # this is intended as a weight to moderate the effect of the impact of other agents defecting
        self.defect_past = []  # list of past defection decisions of the agent
        self.peer_defect = None  # share of peers that defect
        # for now, this is set in the defection_probability_interaction
        self.recency_bias = None  # TODO make this a model level parameter - accounts for how many of the past decisions to include in the current decision
        self.complete_dr_initialization()
        print("initialized defect prob = ", self.defect)
        print("wt_peer = ", self.wt_peer)

        # ----------------------------------------------

        # Initialize Agent attributes from json file car_values.json when creating an Agent
        self.car_values = dict()
        self.battery_capacity = None
        self.car_size = None
        # self.number_of_car_model = None
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
        # self.grid_load = None

        # self.charger_to_charger_trips = self.set_charger_to_charger_trips()
        self.consumption_to_next_charge = None
        self.charging_duration = None
        self.charging_priority = None

        self.base_load = 0
        self.capacity_to_charge = None

        self.final_charging_value = False

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
        with open(CAR_VALUES_PATH) as f:
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

    def set_weight_peer(self):
        np.random.seed(self.seed_value)
        if self.defect:
            wt_peer = self.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # self.random.random()
        else:
            wt_peer = 0
        self.wt_peer = wt_peer

    def set_defection_probability(self):
        if self.defect:
            defect_prob = self.random.randint(0, 1)
        else:
            defect_prob = 0
        self.defect = defect_prob

    def set_recency_bias(self):
        """no of past events that the agent accounts for in the decision at current timestep"""
        self.recency_bias = 5  # TODO set this as a model parameter, do not fix value here.

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

    def complete_dr_initialization(self):
        """
        Run all functions to initialize the attributes for the car agents.
        """
        self.set_defection_probability()
        self.set_weight_peer()
        self.set_recency_bias()

    def load_matching_df(self) -> pd.DataFrame:
        """
        Load the csv file having the matching between median trip length in the mobility file with the battery size
        of the car. Small cars get matched with mobility files having, based on median trip length,
        shorter trips for the time period of the Simulation.
        """
        # this file generated with one time run in match_cars_mobility.py
        # try to load it otherwise it will be generated
        try:
            df = pd.read_csv(MEDIAN_TRIP_LEN_PATH, index_col=0)
        except:
            print("Mobility mapping file has to be generated once.\n"
                  "This might take a while.")
            self.create_matching_file(MEDIAN_TRIP_LEN_PATH)
            df = pd.read_csv(MEDIAN_TRIP_LEN_PATH, index_col=0)
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
        # Set seed value
        random.seed(self.seed_value)
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
        raw_mobility_data = pd.read_csv(file_path,
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

        self.mobility_data['AT_CHARGER'] = self.mobility_data.apply(lambda row: 1 if ((row['CLUSTER'] in [1, 2])
                                                                    and (row['ID_PANELSESSION'] == 0)) else 0, axis=1)

        print("Adding mobility profile of car {} to agent {} ...".format(self.car_id, self.unique_id))
        # self.mobility_data.to_csv("mobility_test_data.csv")
        # self.mobility_data = self.load_mobility_data('mobility_test_data.csv')

    def get_panel_session(self):
        return self.panel_session

    def set_consumption_to_next_charge(self):
        """
        Next trip is not defined by trip number instead it is defined by
        leaving a charger and reaching a new charger (Cluster 1 or 2).
        No public chargers considered here.
        """
        data = self.get_mobility_data()
        data = data.copy()
        # Create groups
        group_label = (data['AT_CHARGER'] != data['AT_CHARGER'].shift(1)).cumsum()
        data['GROUP'] = group_label
        data['GROUP_SHIFTED'] = group_label - 1
        # calculate the sums of the group labels
        group_sums = data.groupby(group_label)['ECONSUMPTIONKWH'].transform('sum')
        # assign them to a column
        data['TRIP_CONSUMPTION'] = group_sums
        # map the group sums to the previous group and fill nan with 0
        mapping = dict(zip(data['GROUP_SHIFTED'], data['TRIP_CONSUMPTION']))
        data['NEXT_TRIP_CONSUMPTION'] = data['GROUP'].map(mapping).fillna(0)

        self.consumption_to_next_charge = data.loc[self.timestamp, 'NEXT_TRIP_CONSUMPTION']

    # def set_consumption_to_next_charge(self):
    #     """
    #     Next trip is not defined by trip number instead it is defined by
    #     leaving a charger and reaching a new charger (Cluster 1 or 2).
    #     No public chargers considered here.
    #     """
    #     id_panel = self.get_panel_session()
    #     mobility_data = self.get_mobility_data()
    #
    #     cluster = self.get_cluster()
    #     # cluster indicates if the car is at home, work or somewhere else
    #     # panel session if the car is driving, ignition, or turning engine off
    #     # cluster 1 = home, cluster 2 = work
    #     if (cluster == 1 and id_panel == 0) or (cluster == 2 and id_panel == 0):
    #         # We are interested in the consumption the next trip has
    #         # A new possibility to charge always occurs when the car is not driving and when the car is at home or work
    #         next_block = (mobility_data['CLUSTER'].ne(1)
    #                       & mobility_data['CLUSTER'].ne(2)
    #                       & mobility_data['ID_PANELSESSION'].ne(0)).idxmax()
    #         print(next_block)
    #         next_trip_consumption = mobility_data.loc[next_block:, 'ECONSUMPTIONKWH'].sum()
    #         # print(next_trip_consumption)
    #         self.consumption_to_next_charge = next_trip_consumption
    #     else:
    #         self.consumption_to_next_charge = 0
    #     print(self.consumption_to_next_charge)
    #     breakpoint()

    def get_consumption_to_next_charge(self):
        return self.consumption_to_next_charge

    def set_charging_duration(self):
        id_panel = self.get_panel_session()
        cluster = self.get_cluster()
        mobility_data = self.get_mobility_data()

        # Find block in the mobility data where id panel is 0 and cluster is 1 or 2
        # Means the car is not driving and is at a home or work location
        if (id_panel == 0 and cluster == 1) or (id_panel == 0 and cluster == 2):
            next_block_start = mobility_data.loc[self.timestamp:, 'ID_PANELSESSION'].eq(0).idxmax()
            next_block_end = mobility_data.loc[next_block_start:, 'ID_PANELSESSION'].ne(0).idxmax()
            charging_time = mobility_data.loc[next_block_start:next_block_end]
            # Calculate the duration
            duration = charging_time.index.max() - charging_time.index.min()
            duration = duration.total_seconds() / 3600
            self.charging_duration = duration
        else:
            self.charging_duration = 0

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
        # to not get battery lvl below 0 only needed if there is consumption more than battery capacity
        new_battery_lvl = max(new_battery_lvl, 0)
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

    def get_charging_efficiency(self):
        return self.charging_eff

    def calc_charging_value(self):
        """
        Function to calculate the real charging value.

        This charging value is in kW and considers:
        - charging power of the car
        - charging power of the charging station
        - empty battery capacity regarding total capacity
        - empty battery capacity regarding soc

        """
        efficiency = self.get_charging_efficiency()

        # battery capacity is e.g. 50 kwh
        empty_battery_capacity = self.empty_battery_capacity()   # kwh
        # calculated in kwh
        possible_soc_capacity = self.empty_battery_capacity_soc()    # kwh

        # Set correct charging power for the car based on cluster
        charging_power_car = self.get_charging_power_car()
        charging_value_car = aux.convert_kw_kwh(kw=charging_power_car)
        # less kwh is reaching the car
        real_charging_value_car = charging_value_car * efficiency / 100

        # Set correct charging power for the station based on cluster
        charging_power_station = self.get_charging_power_station()
        charging_value_station = aux.convert_kw_kwh(kw=charging_power_station)
        # less kwh is reaching the car
        real_charging_value_station = charging_value_station * efficiency / 100

        # TO-DO: Check this value!
        # function claims that charging_value is in kW
        # but it seems to me that the value is in kWh
        charging_value = min(empty_battery_capacity,
                             possible_soc_capacity,
                             real_charging_value_car,
                             real_charging_value_station)

        debug = False
        if debug:
            print("empty_battery_capacity: {} "
                  "possible_soc_capacity {} "
                  "real_charging_value_car {} "
                  "real_charging_value_station {} ".format(empty_battery_capacity,
                                                           possible_soc_capacity,
                                                           charging_value_car,
                                                           charging_value_station))

        return charging_value

    def get_cluster(self):
        """
        Function to return the location of the car.
        1 = Home
        2 = Work
        0 = Everywhere else / Public
        """
        return self.cluster

    def set_charging_power_car(self):
        """
        Can only charge at home or work.
        To set charging to public, replace > else charging_power_car = 0
        with the value the car should be able to charge when stopping at public location.
        """
        cluster = self.get_cluster()
        ac_charging_capacity = self.get_charging_power_ac()
        # dc_charging_capacity = self.get_charging_power_dc()

        # Both are set to ac charging because we consider work has also ac chargers
        if cluster == 1:  # home
            self.charging_power_car = ac_charging_capacity
        elif cluster == 2:  # work
            self.charging_power_car = ac_charging_capacity
        else:
            # Change this, if car should charge everywhere e.g. public charging
            # This could also be dc charging
            self.charging_power_car = 0

    def get_charging_power_car(self):
        return self.charging_power_car

    def set_charging_power_station(self):
        with open(CHARGER_VALUE_PATH) as f:
            charging_stations_dict = json.load(f)

        cluster = self.get_cluster()

        if cluster == 1:
            self.charging_power_station = charging_stations_dict['home']
        elif cluster == 2:
            self.charging_power_station = charging_stations_dict['work']
        else:
            self.charging_power_station = charging_stations_dict['public']

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

        RETURNS: None 

        Priority level for each car: min = 3, max = 9
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

        # TO-DO: range anxiety is always accounted for in the relative need calculation
        relative_need = consumption_next_trip_range_anx / battery_capacity * 100

        if relative_need <= 20:
            prio_next_trip = 1
        elif 20 < relative_need < 80:
            prio_next_trip = 2
        else:
            prio_next_trip = 3

        # in hours
        # FIXME
        # TODO: this should also account for the soc level
        # currently, if very close to departure, priority is set to high
        # but it should not be high if soc is already 100% or soc_limit - that is a parameter, use that here!
        # therefore, add a check for SOC
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
            print('timestamp {},'
                  'soc {}, prio_soc {},'
                  'next_trip {}, prio_next_trip {}, '
                  'charging_duration {}, prio_time {}'.format(self.timestamp,
                                                              self.soc,
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

    def set_final_charging_value(self, value: bool):
        self.final_charging_value = value

    def get_final_charging_value(self) -> bool:
        """ 
        -> bool is a type hint
        i.e. the function is expected to return a bool value
        """
        return self.final_charging_value

    def defection_probability_interaction(self, agents_all):
        """
        function for updating the defection probability of agents 
        """
        total_defect = 0
        ele_vehs_all = []

        # Calculate the share of defecting agents
        for ele_veh in agents_all:
            if isinstance(ele_veh, ElectricVehicle):
                ele_vehs_all.append(ele_veh)
                if ele_veh.defect:
                    total_defect += 1
        ratio_defectors = total_defect/len(ele_vehs_all)

        for e_veh in ele_vehs_all:
            e_veh.peer_defect = ratio_defectors
            e_veh.defect_past.append(e_veh.defect)  # saving past decisions of agent
            e_veh.prob_defect = e_veh.wt_peer*ratio_defectors + (1-e_veh.wt_peer)*np.mean(e_veh.defect_past)
            e_veh.defect = random.random() < e_veh.prob_defect  # returns True with a probability of e_veh.test

    def get_transformer_capacity(self):
        all_agents = self.model.schedule.agents
        transformer_capacity = 0
        # GET ALL TRANSFORMER CAPACITIES FROM ALL TRANSFORMER AGENTS
        for transformer in all_agents:
            if isinstance(transformer, Transformer):
                transformer_capacity += transformer.get_capacity_kw()
        return transformer_capacity

    def get_customer_load(self):
        all_agents = self.model.schedule.agents
        # GET ALL LOAD FROM ALL CUSTOMER AGENTS
        customer_load = 0
        for power_customer in all_agents:
            if isinstance(power_customer, PowerCustomer):
                customer_load += power_customer.get_current_load_kw()
        return customer_load

    def get_total_charging_value(self, ev_agents):
        total_charging_value = 0
        # calculate the total charging values of all car agents in the model
        for charging_value in ev_agents:
            if charging_value is not None:
                total_charging_value += charging_value.get_charging_value()
                # in the above line of code, the get_charging_value simply returns the charging_value that has
                # previously been set in the calc_charging_value function. So here we are simply fetching the
                # charging value. This is also a kWh value
        return total_charging_value

    def interaction_charging_values(self):
        all_agents = self.model.schedule.agents

        transformer_capacity = self.get_transformer_capacity()
        customer_load = self.get_customer_load()

        # Calculate the available capacity to charge EVs
        capacity = transformer_capacity - customer_load

        # all electric vehicle agents
        electric_vehicles_all = []  # all EVs are includeded in this list
        electric_vehicles = []  # only EVs with defect = 0 are included here
        for electric_vehicle in all_agents:
            if isinstance(electric_vehicle, ElectricVehicle):
                # old code commented out
                electric_vehicles_all.append(electric_vehicle)
                # !!! remove those agents that have a defection probability of 1
                if not electric_vehicle.defect:
                    electric_vehicles.append(electric_vehicle)

        total_charging_value = self.get_total_charging_value(electric_vehicles)

        # kw total charging power
        total_charging_power = aux.convert_kw_kwh(kwh=total_charging_value)  # converts the kWh value to kW

        all_priorities = []
        for prio in electric_vehicles:
            all_priorities.append(prio.get_charging_priority())

        # only enter if transformer capacity is exceeded
        # else nothing happens -> the calculated charging_power remains as is for all the car agents
        if total_charging_power > capacity:
            highest_priority = max(all_priorities)
            lowest_priority = min(all_priorities)

            if highest_priority == lowest_priority:  # all cars have the same priority
                distributed = 0     # kw
                while True:
                    # get all agents that are completed / have final_charging_value = True
                    # while loop is ended by a break - no other end
                    completed_charging_agents = []
                    for completed in electric_vehicles:
                        final = completed.get_final_charging_value()
                        # this is to track 'before-interaction' and 'after-interaction' values
                        if final:
                            # after-interaction reduced charging values are set
                            completed_charging_agents.append(completed)

                    # number of not finalized charging values
                    remaining_agents = len(electric_vehicles) - len(completed_charging_agents)
                    # when executed for the first time, this is going to be non-zero

                    available_capacity = capacity - distributed
                    if remaining_agents > 0:
                        charging_power_per_agent = available_capacity / remaining_agents
                        # equal distribution of available capacity across remaining agents
                    else:
                        charging_power_per_agent = 0

                    for ev in electric_vehicles:
                        if not ev.get_final_charging_value():  # after-interaction charging value is not already set ; "not FALSE"
                            # kwh
                            charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                            # kwh
                            new_charging_value = min(charging_value_per_agent, ev.get_charging_value())
                            # kwh, kwh
                            if new_charging_value >= ev.get_charging_value():
                                # ??? - what about the case when new_charging_value<ev.get_charging_value() -> how is distributed updated?
                                # ??? how is the set_final_charging_value set to True?
                                # !!! - this is done in the next few lines (827 onwards)
                                # ??? does this case ever happen? Because just set to minimum before this if statement - happens for equal
                                # ??? why was doing this necessary?
                                ev.revert_charge()  # to effectively go back a timestep and revert charging so that battery level is such
                                # that no charging happened
                                ev.set_charging_value(new_charging_value)
                                ev.set_final_charging_value(True)
                                ev.charge()
                                new_charging_power = aux.convert_kw_kwh(kwh=new_charging_value)
                                distributed += new_charging_power

                    # in the previous lines of code, the interaction only reset the charging values for agents which satisfied the condition
                    # new_charging_value >= ev.get_charging_value()
                    # here, we continue for all other agents. After that, break to get out of the while loop
                    completed_charging_agents_after = []
                    for completed in electric_vehicles:
                        if completed.get_final_charging_value():
                            completed_charging_agents_after.append(completed)
                    if len(completed_charging_agents) == len(completed_charging_agents_after):
                        available_capacity = capacity - distributed
                        remaining_agents = len(electric_vehicles) - len(completed_charging_agents_after)

                        if remaining_agents > 0:
                            charging_power_per_agent = available_capacity / remaining_agents

                            for elec_vehic in electric_vehicles:
                                if not elec_vehic.get_final_charging_value():
                                    # final_charging_value has not been set, so set now.
                                    # !!! - the agents which had new_charging_value<ev.get_charging_value() are set here.
                                    elec_vehic.revert_charge()
                                    charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                                    elec_vehic.set_charging_value(charging_value_per_agent)
                                    elec_vehic.set_final_charging_value(True)
                                    elec_vehic.charge()
                                    new_charging_power = aux.convert_kw_kwh(kwh=charging_value_per_agent)
                                    distributed += new_charging_power

                        break

                # Reset all final charging bools
                for ev in electric_vehicles:
                    ev.set_final_charging_value(False)
            else:
                distributed = 0  # kw
                # Start with the highest priority to charge proceed
                for priority in range(highest_priority, lowest_priority - 1, -1):
                    # get all agents with this priority
                    agents_with_priority = []
                    for ev in electric_vehicles:
                        if priority == ev.get_charging_priority():
                            agents_with_priority.append(ev)

                    # check if some priorities are skipped e.g. 1 car prio 5 other prio 3, 4 is then skipped
                    if len(agents_with_priority) == 0:
                        continue

                    while True:
                        # get all agents that are completed / have final_charging_value = True
                        completed_charging_agents = []
                        for completed in agents_with_priority:
                            final = completed.get_final_charging_value()
                            if final:
                                completed_charging_agents.append(completed)
                        # number of not finalized charging values
                        remaining_agents = len(agents_with_priority) - len(completed_charging_agents)

                        available_capacity = capacity - distributed
                        if remaining_agents > 0:
                            charging_power_per_agent = available_capacity / remaining_agents
                        else:
                            charging_power_per_agent = 0

                        for ev in agents_with_priority:
                            if not ev.get_final_charging_value():
                                # kwh
                                charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                                # kwh
                                new_charging_value = min(charging_value_per_agent, ev.get_charging_value())
                                # kwh, kwh
                                if new_charging_value >= ev.get_charging_value():
                                    ev.revert_charge()
                                    ev.set_charging_value(new_charging_value)
                                    ev.set_final_charging_value(True)
                                    ev.charge()
                                    new_charging_power = aux.convert_kw_kwh(kwh=new_charging_value)
                                    distributed += new_charging_power

                        completed_charging_agents_after = []
                        for completed in agents_with_priority:
                            if completed.get_final_charging_value():
                                completed_charging_agents_after.append(completed)

                        # check if there are more completed agents after loop before
                        # ??? the following comparison does not make sense at all
                        # imo: completed_charging_agents will always be >= completed_charging_agents_after because not all agents
                        # will satisfy the condition of new_charging_value >= ev.get_charging_value() in line 897
                        # BUG: is it possible that there is comparison error in the next line?
                        if len(completed_charging_agents) == len(completed_charging_agents_after):
                            available_capacity = capacity - distributed
                            remaining_agents = len(agents_with_priority) - len(completed_charging_agents_after)
                            if remaining_agents > 0:
                                charging_power_per_agent = available_capacity / remaining_agents

                                for elec_vehic in agents_with_priority:
                                    if not elec_vehic.get_final_charging_value():
                                        elec_vehic.revert_charge()
                                        charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                                        elec_vehic.set_charging_value(charging_value_per_agent)
                                        elec_vehic.set_final_charging_value(True)
                                        elec_vehic.charge()
                                        new_charging_power = aux.convert_kw_kwh(kwh=charging_value_per_agent)
                                        distributed += new_charging_power

                            break

                # Reset all final charging bools
                for ev in electric_vehicles:
                    ev.set_final_charging_value(False)

        # self.defection_probability_interaction(electric_vehicles_all)
        # for e_v in electric_vehicles_all:
        #    print("Defection probability of EV#", e_v.unique_id, " = ", e_v.defect)

    def step(self):
        if self.timestamp is None:
            self.timestamp = self.start_date
        else:
            # Each step add 15 minutes
            #self.timestamp += datetime.timedelta(minutes=15)
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
        self.charge()
        self.set_car_charging_priority()

        if self.charging_algo:
            # Check if the step is done for the last agent in model
            # Start the interaction
            all_agents = self.model.schedule.agents

            all_agents_ids = []
            ev_agents = []
            for electric_vehicle in all_agents:
                if isinstance(electric_vehicle, ElectricVehicle):
                    all_agents_ids.append(electric_vehicle.get_unique_id())
                    ev_agents.append(electric_vehicle)

            # TODO: here add a function to update the defect probability based on other agents' defect probabilities AND some randomness?
            # or should it be called within the interaction function?

            current_agent_id = self.get_unique_id()

            # Check if current agent id is the last id in list of ids of scheduled agents then interact
            if all_agents_ids[-1] == current_agent_id:
                # Calculate how much capacity is available for charging cars after household base load
                self.interaction_charging_values()

                # calculate new defection probability after a defection happened.
                transformer_capacity_after_int = self.get_transformer_capacity()
                customer_load_after_int = self.get_customer_load()
                total_ev_charging_value_after_int = self.get_total_charging_value(ev_agents)
                total_ev_charging_power_after_int = aux.convert_kw_kwh(kwh=total_ev_charging_value_after_int)
                if total_ev_charging_power_after_int > (transformer_capacity_after_int - customer_load_after_int):
                    self.defection_probability_interaction(all_agents)  # only update defection probability if overload happens


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
