import json
import pandas as pd
import auxiliary as aux
import datetime
from mobility_data import MobilityDataAggregator
import mesa
import match_cars_mobility as mcm


class ElectricVehicle(mesa.Agent):
    # list of all picked mobility data to assign them only once
    picked_mobility_data = []

    def __init__(self,
                 unique_id: int,
                 car_model: str,
                 target_soc: float,
                 start_date: str,
                 end_date: str,
                 model=None):
        """
        :param car_model: 'bmw_i3' | 'renault_zoe' | 'tesla_model_3' | 'vw_up' | 'vw_id3' | 'smart_fortwo' | 'hyundai_kona' | 'fiat_500' | 'vw_golf' | 'vw_id4_id5'
        :param target_soc: 0.00 - 1.00, charging happens until target SOC has been reached
        """
        # TODO sort them differently and maybe reduce them drastically
        # insert the super class
        super().__init__(unique_id, model)

        self.sorted_models = None
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
        # self.current_charging = None

        # run this always when creating a car agent
        self.initialize_car_values()

        self.mobility_data = None
        self.start_date = start_date
        self.end_date = end_date
        self.load_mobility_data()

        self.reduction = 0

        self.current_timestamp = None

    def set_timestamp(self, timestamp):
        self.current_timestamp = timestamp

    def initialize_car_values(self):
        # load car values from JSON file in directory
        with open('../input/car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        self.battery_capacity = car_dict[self.car_model]["battery_capacity"]
        self.number_of_car = car_dict[self.car_model]["number"]
        self.charging_power_home = car_dict[self.car_model]["charging_power_ac"]
        self.charging_power_word = car_dict[self.car_model]["charging_power_dc"]

        if self.car_size is None:
            self.sorted_models = sorted(car_dict, key=lambda x: car_dict[x]['battery_capacity'])
            self.car_size = self.sorted_models.index(self.car_model)

    def load_mobility_data(self):
        # this file generated with one time run in match_cars_mobility.py
        file_name_median_trip_len = '../median_trip_length.csv'
        try:
            df = pd.read_csv(file_name_median_trip_len, index_col=0)
        except:
            print("Mobility mapping file needs to be generated once.\n"
                  "This might take a while.")
            directory_path = aux.get_directory_path()
            no_clusters = len(self.sorted_models)
            mcm.create_median_trip_length_file(directory_path=directory_path,
                                               start_date=self.start_date,
                                               end_date=self.end_date,
                                               no_deciles=no_clusters,
                                               file_name=file_name_median_trip_len)
            df = pd.read_csv(file_name_median_trip_len, index_col=0)

        # check all already picked 'car_ids' and filter them out of df
        df = df.loc[~df['car_id'].isin(ElectricVehicle.picked_mobility_data)]
        # if the dataframe is empty clean the picked cars and start again / mobility data can be picked twice
        if df.empty:
            ElectricVehicle.picked_mobility_data.clear()
            df = df.loc[~df['car_id'].isin(ElectricVehicle.picked_mobility_data)]
        # get the closest number in decile_label to car_size
        closest_number = min(df['decile_label'], key=lambda x: abs(x - self.car_size))

        # Get a list of indexes where decile_label is equal to closest_number
        matching_indexes = df.index[df['decile_label'] == closest_number].tolist()
        import random
        # Get a random index from the matching indexes
        random_index = random.choice(matching_indexes)
        # # get first index where condition is met
        # matching_indecis = df.loc[df['decile_label'] == closest_number].index[0]
        # find the car id
        random_car_id = df.loc[random_index, 'car_id']
        # add the car id to already picked ids
        ElectricVehicle.picked_mobility_data += [random_car_id]
        self.car_id = random_car_id

        # TEST True = Set local directory for mobility data
        # Load correct mobility file
        file_path = aux.create_file_path(random_car_id, test=True)

        # read only used columns to speed up data reading
        raw_mobility_data = pd.read_csv(file_path, usecols=['TIMESTAMP', 'TRIPNUMBER', 'DELTAPOS', 'CLUSTER', 'ECONSUMPTIONKWH', 'ID_PANELSESSION', 'ID_TERMINAL'])
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
        #TODO Replace self.get_mobility_data() with df
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
    def calculate_battery_level(self, charging_efficiency=0.95, reduced=False):

        if self.battery_level is None:
            self.battery_level = self.battery_capacity

        self.calc_soc()

        self.consumption = self.mobility_data.loc[self.current_timestamp, 'ECONSUMPTIONKWH']

        charging = self.charging_possible(self.soc,
                                          self.target_soc,
                                          self.plugged_in,
                                          self.consumption)

        if charging is None:
            charging = True

        if not charging:
            potential_battery_level = self.battery_level - self.consumption
            if potential_battery_level < 0:
                new_consumption_value = min(self.battery_level, self.consumption - self.battery_level)
                self.battery_level -= new_consumption_value
            else:
                self.battery_level -= self.consumption

            self.charging_value = 0
        else:  # charging
            # TODO charging power home should be charging_power ->
            #  min(charging_power_home, charging_power_station)
            potential_battery_level = self.battery_level + self.charging_power_home
            if potential_battery_level >= self.battery_capacity:
                over_charged_value = potential_battery_level - self.battery_capacity
                if not reduced:
                    self.charging_value = max(0, self.charging_power_home - over_charged_value)
                    self.battery_level += self.charging_value
                    # self.load_curve.append(new_charging_value)
                else:
                    # if reduction happened just take the self.charging_value from class
                    self.battery_level += self.charging_value
            else:
                if not reduced:
                    # check for target soc and reduce charging power according to it
                    self.charging_value = self.target_soc * self.battery_capacity - self.battery_level
                    self.charging_value = max(self.charging_value, self.charging_power_home)
                    self.battery_level += self.charging_value
                else:   # just take self.charging_value if already reduced
                    self.battery_level += self.charging_value

    def set_soc(self):
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

    def get_current_charging_value(self):
        return self.charging_value

    def get_soc(self):
        return self.soc




    def update_charging_value(self):
        all_agents = self.model.schedule.agents    # get all agents of the model
        car_agents = [agent for agent in all_agents if isinstance(agent, ElectricVehicle)]    # filter all agents by electricvehicles
        car_agents_charging_values = [agent.charging_value for agent in car_agents]       # get all charging values of all agents
        car_agents_charging_values_total = sum([x for x in car_agents_charging_values if x is not None])     # calculate the total charging values of all agents in the model
        # TODO max_capacity retrieved by a different agent
        max_capacity = 25
        num_charging_cars = len([value for value in car_agents_charging_values if value is not None and value != 0])  # filter what agents are charging at that timestamp

        if car_agents_charging_values_total > max_capacity:  # check if the total_charging_power is above the max_capacity
            exceeding_charging_value = car_agents_charging_values_total - max_capacity    # calc the exceeding energy
            reduction_per_agent = round(exceeding_charging_value / num_charging_cars, 1)   # calc the reduction per charging agent and round the result

            for agent in car_agents:    # update in all car agents the charging value
                if agent.charging_value is not None and agent.charging_value != 0:
                    agent.charging_value = max(0, agent.charging_value - reduction_per_agent)
                    agent.reduction = reduction_per_agent

    def check_reduction(self):
        return self.reduction

    def step(self):
        if self.current_timestamp is None:
            self.current_timestamp = self.start_date
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
        else:
            # each step add 15 minutes
            self.current_timestamp = self.current_timestamp + datetime.timedelta(minutes=15)

        self.set_plug_in_status()
        # TODO FUNCTION TO CALC CCHARGING VALUE FIRRST
        # self.charging_value += self.reduction
        self.reduction = 0
        # calculate battery lvl
        self.calculate_battery_level()

        # # update charging value
        self.update_charging_value()
        # # self.update_other_values()
        # self.calculate_battery_level(reduced=True)

        # self.calculate_battery_level(reduced=True)





if __name__ == "__main__":
    agent = ElectricVehicle(1, "renault_zoe", 1.0, '2008-07-13', '2008-07-27')