import json
import glob
import timeit
import random
from mobility_data import MobilityDataAggregator

import pandas as pd
import numpy as np


def match_cars_mobility_data():
    # Load the car models
    with open('car_values.json') as f:
        car_dict = json.load(f)

    sorted_models = sorted(car_dict, key=lambda x: car_dict[x]['battery_capacity'])

    return sorted_models



def label_mobility_data(df, no_deciles: int):
    # drop entries in dataframe without any trip length
    df = df.dropna()
    # Label the car_ids with 10 different numbers in ascending order (1 = short trip length, 10 = long trip length
    # 10 because we have 10 different cars
    df['decile_label'] = pd.qcut(df['median_trip_length'], q=no_deciles, labels=False, duplicates='drop')
    df = df.sort_values('decile_label')
    df = df.reset_index(drop=True)
    df.to_csv('median_trip_length.csv')
    return df


def median_trip_length(df, car_id):
    len_dict = {}
    trip_df = df.groupby('TRIPNUMBER').sum()
    med_trip_len = trip_df['DELTAPOS'].median()
    len_dict[car_id] = med_trip_len
    print('Successfully added entry for car {} to len dict.'.format(car_id))
    return len_dict


def create_median_trip_length(directory_path, id_segmentation_df):
        start = timeit.default_timer()
        # retrieve all csv_files in the directory path
        csv_files = glob.glob(directory_path + "/*.csv")

        len_dict = {}
        # Loop thorugh all csv files for
        for file in csv_files:
            time_loop_iteration_start = timeit.default_timer()   # start timer
            mobility_data = pd.read_csv(file)
            unique_id = mobility_data['ID_TERMINAL'].unique()[0]    # get unique car id
            # Check if car_id is in one of the private car clusters (2, 3, 6, 8 represent commercial)
            if id_segmentation_df.loc[id_segmentation_df['id'] == unique_id, 'CLUSTER'].values[0] in [1, 4, 5, 7]:
                data = MobilityDataAggregator(mobility_data)
                mobility_data = data.prepare_mobility_data(starting_date='2008-07-13', num_days=7)
                # calculate the median trip length and store it in a dict
                new_dict = median_trip_length(mobility_data, unique_id)
                len_dict.update(new_dict)
            time_loop_iteration_end = timeit.default_timer()
            print('Time: ', time_loop_iteration_end - time_loop_iteration_start, ' secs.')

        # sort the dict according to the median trip length and save it afterwards
        sorted_dict = dict(sorted(len_dict.items(), key=lambda item: item[1]))
        sorted_dict_df = pd.DataFrame(sorted_dict, index=[0])
        sorted_dict_df = sorted_dict_df.T
        sorted_dict_df = sorted_dict_df.reset_index()
        sorted_dict_df.columns = ['car_id', 'median_trip_length']
        sorted_dict_df.to_csv("median_trip_length.csv")

        stop = timeit.default_timer()
        print('Total Runtime: ', stop - start)


def generate_cars_according_to_dist(number_of_agents):
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

    car_models = np.random.choice(cars, size=number_of_agents, p=distribution)
    # print(len(car_names), "car names generated.")

    return car_models


def number_of_each_car(car_model, car_models_list):
    car_series = pd.Series(car_models_list)
    car_counts = car_series.value_counts()
    if car_model in car_counts.index:
        return car_counts[car_model]
    else:
        return 0


if __name__ == '__main__':
    directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
    id_segmentation = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\segmented_ids_07112021.xlsx"
    id_segmentation_df = pd.read_excel(id_segmentation)
    id_segmentation_df = id_segmentation_df[['CLUSTER', 'id']]

    # create a list with car types sorted to their battery capacity
    sorted_models = match_cars_mobility_data()
    # count the length of the list to figure out how many clusters needed
    no_clusters = len(sorted_models)

    # read the dataframe with car_id and trip_length
    try:
        df = pd.read_csv('median_trip_length.csv', index_col=0)
    except FileNotFoundError:
        create_median_trip_length(directory_path, id_segmentation_df)
        df = pd.read_csv('median_trip_length.csv', index_col=0)
    df = label_mobility_data(df, no_deciles=no_clusters)
    number_of_cars = len(df)

    car_models = generate_cars_according_to_dist(number_of_cars)
    print("Created a list containing following car models: ")
    for model in set(car_models):
        number = number_of_each_car(car_model=model, car_models_list=car_models)
        print("Number: {} | Model: {} | Size: {}".format(number, model, sorted_models.index(model)))

    # TODO
    # REFACTOR VERY COMPLICATED
    # SHOULD RATHER CHECK FOR THE CLOSEST VALUE NOT INCREASE IT FIRST
    # AND THEN DECREASE IT AFTER

    min_size = 0
    max_size = 9

    # for all car_models list of ~ 700 cars
    for model in car_models:
        # TODO Create CarAgent here and then load the correct mobility data in the agent
        # get the car size for that model
        size = sorted_models.index(model)
        # As long as the car size of that model is above 0
        while size >= min_size:
            # Get a list of all indices of the dataframe (holding trip length of mobility data)
            indices = df.index[df['decile_label'] == size].tolist()
            # if the list is empty because no match between car size and trip length decile
            if not indices:
                # Reduce the car size by one
                size -= 1
            else:
                # If car size matches trip length size chose a random entry of the indices
                random_index = random.choice(indices)
                # Retrieve the car_id for the random index
                car_id = df.loc[random_index, 'car_id']
                # Remove it from the dataframe that it cannot be chosen 2 times
                df = df.drop(random_index)
                break
        else:
            size += 1
            while size <= max_size:
                indices = df.index[df['decile_label'] == size].tolist()
                if not indices:
                    size += 1
                else:
                    random_index = random.choice(indices)
                    car_id = df.loc[random_index, 'car_id']
                    df = df.drop(random_index)
                    break

        # create the file path to read the mobility data for the car model
        directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
        file_name = '\quarterly_simulation_' + str(car_id) + '.csv'
        file = directory_path + file_name
        df_car = pd.read_csv(file)
