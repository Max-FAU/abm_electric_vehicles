import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import json
# model


# calculate energy consumption for every trip (distance travelled car, and what road)
# input for that is the trip file

# create electric vehicle with battery
# assign battery capacity

# charging possibility
# assign charging power
# assign charging connection (??)

# Initialization of the model
# Retrieve the EV Agent data
# mobility profile
# EV characteristics
# risk attitude?

# retrieve charging stations

# workflow for one EV agent
# every 15 minute step
# can the car charge at that timestep? -> YES -> NO
# is it the first time step the car can charge?
# if yes then check if there is an available charging station
# do the charging decision according to the plug-in behavior
# Does the EV agent intend to charge?
# if yes update SOC
# if no go to the next timestep
# if no update the SOC of the car


# calculation of SOC while driving // assumption is the SOC drops linearly with travel distance
# taken from related work gschwendtner paper
def clac_soc(data,
             battery_capacity,
             consumption_row,
             start=True):

    soc_list = []
    i = 0
    # calculate the SOC for every row because it needs SOC from previous row
    for index, row in data.iterrows():
        if start:
            soc_list += [100]
            start = False
        else:
            soc_list += [(soc_list[i - 1] / 100 * battery_capacity - row[consumption_row]) / battery_capacity * 100]
            i += 1

    col_name = consumption_row + '_SOC'
    data[col_name] = soc_list
    return data

    # # soc_start at the start of the dwell time
    # # soc_end at the end of the previous dwell time
    # soc_start = soc_end - consumption_ev * distance_travelled / battery_capacity
    # print(soc_start)
    #
    # electricity_battery = capacity - consumption * 100 / battery_capacity


# initialize electric vehicle
def init_car_agent(name, battery_capacity=None):
    # battery capacity for car name, min_value, expected_value, max_value in kWh
    car_dict = {
        "dummy": {
            "min_value": 30,
            "expected_value": 50,
            "max_value": 100
        }
    }

    return chose_values(name=name,
                        input_dict=car_dict,
                        input_values=battery_capacity)


def chose_values(name, input_dict, input_values):
    if input_values == 'min':
        return input_dict[name]['min_value']
    if input_values == 'expected':
        return input_dict[name]['expected_value']
    if input_values == 'max':
        return input_dict[name]['max_value']
    return input_dict[name]


# calculate the consumption of electric vehicle
def load_consumption_dict(label, values=None):
    # energy consumption ev agents all values are in kwh per 100 km
    # min_value, expected_value, max_value
    # changed rural to extra urban (außerorts)
    # deleted suburban because it is similar to urban this fits our initial labelling
    # "suburban": {"min_value": 14, "expected_value": 19, "max_value": 24},
    consumption_dict = {
        "urban": {"min_value": 15,
                  "expected_value": 20,
                  "max_value": 25},
        "extra urban": {"min_value": 12.5,
                        "expected_value": 17.5,
                        "max_value": 22.5},
        "highway": {"min_value": 25,
                    "expected_value": 30,
                    "max_value": 35}
    }

    return chose_values(name=label,
                        input_dict=consumption_dict,
                        input_values=values)


def replace_location_panelsession(mobility_data):
    location_type = {0: "ignition", 1: "driving", 2: "engine turn-off"}
    mobility_data['ID_LOCATIONTYPE'] = mobility_data['ID_LOCATIONTYPE'].replace(location_type)
    panel_session = {0: "urban", 1: "highway", 2: "extra urban"}
    mobility_data['ID_PANELSESSION'] = mobility_data['ID_PANELSESSION'].replace(panel_session)
    return mobility_data


def calc_consumption(data):
    # get labels and replace ints with strings in mobility data
    data = replace_location_panelsession(data)

    # create list to store consumption
    consumption_list = []
    for index, row in data.iterrows():
        # load the current consumption based on street type
        consumption_expected_value = load_consumption_dict(label=row['ID_PANELSESSION'],
                                                           values='expected')
        consumption_per_m = consumption_expected_value / 100 / 1000  # per 1 km / per 1 meter

        consumption_list += [consumption_per_m * row['DELTAPOS']]

    data['CONSUMPTION'] = consumption_list
    return data


def create_df_limited_time(data: pd.DataFrame, date: str, days: int):
    start_date = pd.to_datetime(date)
    end_date = start_date + timedelta(days=days)
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    data = data[(data['TIMESTAMP'] > start_date) &
                                  (data['TIMESTAMP'] < end_date)]

    return data


def chose_plug_in_behavior():
    # whenever possible
    # 1 x a week
    # 3 x a week
    pass


def choose_charging_scheme():
    # uncontrolled
    # flat
    # off-peak / uncontrolled
    # off-peak / flat
    pass


def aggregate_mobility_data_15_min_steps(data, average=True, sum=True):
    # check if car is moving in the next 15 min or not
    # calculate the total energy demand in that 15 minutes
    df = pd.DataFrame(data)
    # convert timestamp to index
    df.set_index('timestamp', inplace=True)

    # resample to 15-minute intervals
    if average:
        df_resampled = df.resample('15T').mean()
    else:
        pass

    # resample to 15-minute intervals
    if sum:
        df_resampled = df.resample('15T').sum()
        # take first  //
    else:
        df_resampled = df
    return df_resampled


def get_delay_before_charging():
    # introduce the delay before starting to charge
    # 15 minutes
    pass


def read_json_relevant_columns():
    with open('../relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns['relevant_columns']


def prepare_mobility_data(data, trip_no=None, time_filter=True):
    # keep only relevant columns specified in json
    data = data[relevant_columns]

    # filter mobility data only for specific trip if trip number is given
    if trip_no is not None:
        data = data[data['TRIPNUMBER'] == trip_no]

    # keep only data for 2 weeks if time filter is set to TRUE
    if time_filter:
        data = create_df_limited_time(data=data,
                                      date='2008-07-01 00:00:00',
                                      days=14)

    return data


if __name__ == '__main__':
    relevant_columns = read_json_relevant_columns()
    mobility_data = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    mobility_data = pd.read_csv(mobility_data)

    mobility_data = prepare_mobility_data(data=mobility_data,
                                          trip_no=2,
                                          time_filter=False)

    # home = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\Parking_Cluster_1_LatLongMODES_Home.csv"
    # work = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\Parking_Cluster_2_LatLongMODES_Work.csv"
    # home = pd.read_csv(home)

    # get car variables for expected scenario
    car_expected_value = init_car_agent(name='dummy',
                                        battery_capacity='expected')

    # calculate the consumption based on mobility_data
    mobility_data = calc_consumption(data=mobility_data)
    mobility_data.index = mobility_data['TIMESTAMP']

    # # show plot of consumption of ev
    # mobility_data['CONSUMPTION'].plot.line()
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()

    # # show plot of SOC of ev
    mobility_data_with_soc_own = clac_soc(mobility_data,
                                          consumption_row='CONSUMPTION',
                                          battery_capacity=car_expected_value)

    mobility_data_with_soc_prakhar = clac_soc(mobility_data,
                                              consumption_row='ECONSUMPTIONKWH',
                                              battery_capacity=car_expected_value)

    # mobility_data_with_soc_own['SOC'].plot.line()
    # mobility_data_with_soc_prakhar['SOC'].plot.line()
    both_one_plot = pd.DataFrame({'own': mobility_data_with_soc_own['CONSUMPTION_SOC'],
                                  'prakhar': mobility_data_with_soc_prakhar['ECONSUMPTIONKWH_SOC']})
    print(both_one_plot)
    both_one_plot.plot.line()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


