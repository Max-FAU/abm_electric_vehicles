import json
import glob
import pandas as pd
from tqdm import tqdm
from mobility_data import MobilityDataAggregator
from project_paths import MOBILITY_DATA_DIRECTORY_PATH, INPUT_PATH


def read_json_config(keyword):
    with open('input/relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns[keyword]


def set_print_options():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)


def get_directory_path(test=False) -> str:
    directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
    # directory_path = r"J:\Max_Mobility_Profiles\quarterly_simulation"
    if test:
        directory_path = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data"
        # directory_path = r"J:\Max_Mobility_Profiles\quarterly_simulation"
    return directory_path


def create_file_path(car_id, test=False) -> str:
    # create the file path to read the mobility data for the car model
    directory_path = get_directory_path(test=test)
    if test:
        directory_path = get_directory_path(test=test)
        car_id = 80
    file_name = 'quarterly_simulation_' + str(car_id) + '.csv'
    file_name = MOBILITY_DATA_DIRECTORY_PATH / file_name
    return file_name


def median_trip_length(df, car_id) -> dict:
    # Create a dict with key 'car_id' and value 'med_trip_len'
    len_dict = {}
    trip_df = df.groupby('TRIPNUMBER').sum()
    med_trip_len = trip_df['DELTAPOS'].median()
    len_dict[car_id] = med_trip_len
    print('Successfully calculated median trip length for car {}.'.format(car_id))
    return len_dict


def is_private_car(unique_id: int):
    """Load the json which indicates if the car is a private car."""
    with open(INPUT_PATH / 'private_cars.json') as f:
        private_cars = json.load(f)

    car_ids = []
    for entry in private_cars:
        car_ids.append(entry['id'])

    if unique_id in car_ids:
        return True
    else:
        return False


def label_mobility_data(df, no_deciles: int):
    # drop entries in dataframe without any trip length
    df = df.dropna()
    # Label the car_ids with 10 different numbers in ascending order (1 = short trip length, 10 = long trip length
    # 10 because we have 10 different cars
    df = df.copy()
    df['decile_label'] = pd.qcut(df['median_trip_length'], q=no_deciles, labels=False, duplicates='drop')
    df = df.sort_values('decile_label')
    df = df.reset_index(drop=True)
    return df

# TODO Always check min date with max date and create new mapping file if these dates does not match with simulation
# TODO start and end date
def create_median_trip_length_file(directory_path,
                                   start_date,
                                   end_date,
                                   no_deciles,
                                   file_name):
    # retrieve all csv_files in the directory path to loop
    csv_files = glob.glob(directory_path + "/*.csv")
    len_dict = {}

    # Loop through all csv files with mobility data
    for file in tqdm(csv_files, leave=False):
        id_to_check = int(file.split("_")[-1].replace('.csv', ''))
        if is_private_car(unique_id=id_to_check):
            try:
                mobility_data = pd.read_csv(file, usecols=['TIMESTAMP', 'TRIPNUMBER', 'DELTAPOS', 'CLUSTER', 'ECONSUMPTIONKWH', 'ID_PANELSESSION', 'ID_TERMINAL'])
                # Create the dataframe for short time
                data = MobilityDataAggregator(mobility_data, start_date, end_date)
                # calculate the median trip length and store it in a dict
                median_trip_dict = median_trip_length(data.get_processed_df(), id_to_check)
                # Append the car_id with median trip length to dict
                len_dict.update(median_trip_dict)
            except:
                print("Skipped calculation for car id {}.".format(id_to_check))
        else:
            print("No calculation for car id {}, only private cars considered.".format(id_to_check))

    # sort the dict according to the median trip length and save it afterwards
    sorted_dict = dict(sorted(len_dict.items(), key=lambda item: item[1]))
    sorted_dict_df = pd.DataFrame(sorted_dict, index=[0])
    sorted_dict_df = sorted_dict_df.T
    sorted_dict_df = sorted_dict_df.reset_index()
    sorted_dict_df.columns = ['car_id', 'median_trip_length']

    sorted_dict_df = label_mobility_data(sorted_dict_df, no_deciles)
    sorted_dict_df.to_csv(file_name)


def convert_kw_kwh(kw=None, kwh=None):
    if kw is not None:
        kwh = kw / 4  # Convert kW to kWh
        return kwh
    elif kwh is not None:
        kw = kwh * 4  # Convert kWh to kW
        return kw
    else:
        raise ValueError("Either 'kw' or 'kwh' must be provided.")


if __name__ == '__main__':

    directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
    # start_date = self.start_date
    # end_date = self.end_date
    start_date = '2008-07-13'
    end_date = '2008-07-14'
    create_median_trip_length_file(directory_path, start_date, end_date, no_deciles=10, file_name='test.csv')
