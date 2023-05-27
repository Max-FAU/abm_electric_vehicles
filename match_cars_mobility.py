import json
import glob
from mobility_data import MobilityDataAggregator
import pandas as pd
from tqdm import tqdm


def sort_cars_size():
    # Load the car models
    with open('input/car_values.json') as f:
        car_dict = json.load(f)
    # sort the model according to their battery capacity to later match them
    # the index represents the rank where 0 is small and 9 is large
    sorted_models = sorted(car_dict, key=lambda x: car_dict[x]['battery_capacity'])

    return sorted_models


def median_trip_length(df, car_id):
    # Create a dict holding the median trip length for a car id, this dict will be appended to
    # a large dict holding all car_ids later
    len_dict = {}
    trip_df = df.groupby('TRIPNUMBER').sum()
    med_trip_len = trip_df['DELTAPOS'].median()
    len_dict[car_id] = med_trip_len
    print('Successfully calculated median trip length for car {}.'.format(car_id))
    return len_dict


def is_private_car(unique_id: int):
    with open('input/private_cars.json') as f:
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


def create_median_trip_length_file(directory_path,
                                   start_date,
                                   end_date,
                                   no_deciles,
                                   file_name):
    # retrieve all csv_files in the directory path to loop
    csv_files = glob.glob(directory_path + "/*.csv")
    len_dict = {}

    # Loop through all csv files with mobility data
    for file in tqdm(csv_files):
        id_to_check = int(file.split("_")[-1].replace('.csv', ''))
        if is_private_car(unique_id=id_to_check):
            try:
                mobility_data = pd.read_csv(file, usecols=['TIMESTAMP', 'TRIPNUMBER', 'DELTAPOS', 'CLUSTER', 'ECONSUMPTIONKWH', 'ID_PANELSESSION', 'ID_TERMINAL'])
                # Create the dataframe for short time
                data = MobilityDataAggregator(mobility_data, start_date, end_date)
                # calculate the median trip length and store it in a dict
                median_trip_dict = median_trip_length(data.df_processed, id_to_check)
                # Append the car_id with median trip length to dict
                len_dict.update(median_trip_dict)
            except:
                print("Skipped calculation for car with {}.".format(id_to_check))
        else:
            print("No calculation for car {}, only private cars considered.".format(id_to_check))

    # sort the dict according to the median trip length and save it afterwards
    sorted_dict = dict(sorted(len_dict.items(), key=lambda item: item[1]))
    sorted_dict_df = pd.DataFrame(sorted_dict, index=[0])
    sorted_dict_df = sorted_dict_df.T
    sorted_dict_df = sorted_dict_df.reset_index()
    sorted_dict_df.columns = ['car_id', 'median_trip_length']

    sorted_dict_df = label_mobility_data(sorted_dict_df, no_deciles)
    sorted_dict_df.to_csv(file_name)


if __name__ == '__main__':
    directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
    # start_date = self.start_date
    # end_date = self.end_date
    start_date = '2008-07-13'
    end_date = '2008-07-14'
    # create a list with car types sorted to their battery capacity
    sorted_models = sort_cars_size()
    # count the length of the list to figure out how many clusters needed
    no_clusters: int = len(sorted_models)
    #
    create_median_trip_length_file(directory_path, start_date, end_date, no_deciles=no_clusters, file_name='test.csv')
