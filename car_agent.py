import json
import pandas as pd
import helper
from mobility_data import MobilityDataAggregator
import mesa
import random


class ElectricVehicle(mesa.Agent):
    # list of all picked mobility data to assign them only once
    picked_mobility_data = []

    def __init__(self, unique_id, model: str, target_soc: float, start_date: str, end_date: str):
        """
        :param model: 'bmw_i3' | 'renault_zoe' | 'tesla_model_3' | 'vw_up' | 'vw_id3' | 'smart_fortwo' | 'hyundai_kona' | 'fiat_500' | 'vw_golf' | 'vw_id4_id5'
        :param target_soc: 0.00 - 1.00, charging happens until target SOC has been reached
        """
        self.unique_id = unique_id
        self.model = model
        self.car_size = None

        self.battery_capacity = None
        self.battery_level = None
        assert isinstance(target_soc, float), "Target SOC must be a float."
        self.target_soc = target_soc
        self.soc = None

        self.location = None

        self.anxiety_factor = 1.5
        self.range_anxiety = None

        self.plugged_in = None
        self.current_charging = None

        # run this always when creating a car agent
        self.__initialize_car_values()

        self.mobility_data = None
        self.start_date = start_date
        self.end_date = end_date
        self.__load_mobility_data()

        #TODO Check if this work, I do not want to create the agent with different timestamp
        self.current_timestamp = None

        # self.load_curve = []
        # self.battery_level_curve = []
        # self.soc_curve = []

    def set_timestamp(self, timestamp):
        self.current_timestamp = timestamp

    def __initialize_car_values(self):
        # load car values from JSON file in directory
        with open('car_values.json') as f:
            car_dict = json.load(f)

        # retrieve and set car values
        self.battery_capacity = car_dict[self.model]["battery_capacity"]
        self.number_of_car = car_dict[self.model]["number"]
        self.charging_power_home = car_dict[self.model]["charging_power_home"]
        self.charging_power_word = car_dict[self.model]["charging_power_work"]

        if self.car_size is None:
            sorted_models = sorted(car_dict, key=lambda x: car_dict[x]['battery_capacity'])
            self.car_size = sorted_models.index(self.model)

    def __load_mobility_data(self):
        df = pd.read_csv('median_trip_length.csv', index_col=0)

        min_car_size = 0
        max_car_size = 9

        potential_car_ids = df.index[df['decile_label'] == self.car_size].tolist()
        potential_car_ids = [x for x in potential_car_ids if x not in ElectricVehicle.picked_mobility_data]

        random_car_index = random.choice(potential_car_ids)
        random_car_id = df.loc[random_car_index, 'car_id']
        ElectricVehicle.picked_mobility_data += [random_car_id]

        file_path = helper.create_file_path(random_car_id, test=True)

        raw_mobility_data = pd.read_csv(file_path)
        data_aggregator = MobilityDataAggregator(raw_mobility_data=raw_mobility_data,
                                                 start_date=self.start_date,
                                                 end_date=self.end_date)

        self.mobility_data = data_aggregator.df_processed

    def _set_unique_id(self, id_col='ID_TERMINAL'):
        unique_id = self.mobility_data[id_col].unique()
        if len(unique_id) > 1:
            raise Exception('More than one IDs in id_col.')
        self.unique_id = unique_id[0]


    def set_plug_in_status(self, df):

        panel_session = df.loc[self.current_timestamp, 'ID_PANELSESSION']
        current_index = df.index.get_loc(self.current_timestamp)

        # first step for 15 minute buffer, since 1 timestep is 15 mins away
        if self.plugged_in is None:  # first time
            previous_index = current_index
        else:
            previous_index = current_index - 1

        previous_timestamp = df.index[previous_index]
        previous_panel_session = df.loc[previous_timestamp, 'ID_PANELSESSION']

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

    def calculate_battery_level(self, df, charging_efficiency=0.95):

        if self.battery_level is None:
            self.battery_level = self.battery_capacity

        self._calc_soc()

        consumption = df.loc[self.current_timestamp, 'ECONSUMPTIONKWH']

        charging = self.charging_possible(self.soc,
                                          self.target_soc,
                                          self.plugged_in,
                                          consumption)

        if not charging:
            potential_battery_level = self.battery_level - consumption
            if potential_battery_level < 0:
                new_consumption_value = min(self.battery_level, consumption - self.battery_level)
                self.battery_level -= new_consumption_value
            else:
                self.battery_level -= consumption
            # print("consumption")
        elif charging:
            # TODO charging power home should be charging_power -> min(charging_power_home, charging_power_station)
            potential_battery_level = self.battery_level + self.charging_power_home
            if potential_battery_level >= self.battery_capacity:
                over_charged_value = potential_battery_level - self.battery_capacity
                new_charging_value = max(0, self.charging_power_home - over_charged_value)
                self.battery_level += new_charging_value
                # self.load_curve.append(new_charging_value)
            else:
                # check for target soc and reduce charging power according to it
                charging_value = self.target_soc * self.battery_capacity - self.battery_level
                charging_value = max(charging_value, self.charging_power_home)
                self.battery_level += charging_value
            # self.load_curve.append(self.charging_power)
        # print("charging")
        else:
            # print("Car is plugged in with consumption.")
            pass

        # self.battery_level_curve.append(self.battery_level)
        # self._calc_soc()
        print(self.battery_level)
        # return self.battery_level

    def _calc_soc(self):
        # Calculate the state of charge (SoC)
        self.soc = self.battery_level / self.battery_capacity

        # self.soc_curve.append(self.soc)

    def next_trip_needs(self, df):
        """ anxiety factor 1.5 """
        last_trip = max(df['TRIPNUMBER'])
        next_trip = df.loc[self.current_timestamp, 'TRIPNUMBER'] + 1
        consumption_trips = df.groupby('TRIPNUMBER')['ECONSUMPTIONKWH'].sum()
        if next_trip <= last_trip:
            consumption_next_trip = consumption_trips.loc[next_trip]
        else:
            consumption_next_trip = consumption_trips.loc[next_trip - 1]

        self.range_anxiety = consumption_next_trip * self.anxiety_factor

    def step(self):
        self.set_plug_in_status(self.mobility_data)
        self.calculate_battery_level(self.mobility_data)


# class ElectricVehicleFlatCharge(ElectricVehicle):
#     def __init__(self, model, **params):
#         super().__init__(model)
#         self.max_power = 3.7
#         self.min_power = 1.22

# TODO
# Implement flat charging calculation
# implement soc to stop charging
# check each timestamp if soc already reached

# self.min_soc xxxx
# self.range_anxiety


if __name__ == '__main__':
    path = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    raw_data = pd.read_csv(path)

    # initialize car object and retrieve expected battery capacity value
    bmw_i3 = ElectricVehicle(model='bmw_i3',
                             target_soc=0.8,
                             start_date='2008-07-18 00:00:00',
                             end_date='2008-07-20 00:00:00')

    # mobility_data = MobilityDataAggregator(raw_data,
    #                                        start_date='2008-07-18 00:00:00',
    #                                        end_date='2008-07-20 00:00:00')

    # for timestamp in mobility_data.df_processed.index:
    #     # print(timestamp)
    #     # bmw_i3.next_trip_needs(mobility_data, timestamp)
    #     bmw_i3.set_timestamp(timestamp=timestamp)


    # helper.set_print_options()
    #
    #
    # print(mobility_data)

# TODO build new class with results, access results easier and write functions to aggregate them
# TODO aggregated mobility file - read it whole - store maybe as dict
# TODO have timestep as key and consumption as values

# TODO calculate how much energy the next trip needs
# TODO implement the check for SOC, charge only until SOC has been reached
# TODO implement charging only between 18:00 and 06:00
# TODO implement charging in flat manner, means calculate the time the car stands
# TODO divide the charging power by the time the car has to charge
