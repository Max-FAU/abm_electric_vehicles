import json
import pandas as pd


def read_json_config(keyword):
    with open('relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns[keyword]


def aggregation_mode():
    """Helper function to find the mode"""
    return lambda x: x.value_counts().index[0]


def set_print_options():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)


def create_file_path(car_id, test=False) -> str:
    # create the file path to read the mobility data for the car model
    directory_path = r"D:\Max_Mobility_Profiles\quarterly_simulation"
    if test:
        directory_path = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data"
        car_id = 80
    file_name = '\quarterly_simulation_' + str(car_id) + '.csv'
    file_name = directory_path + file_name
    return file_name


if __name__ == '__main__':
    file_path = read_json_config('relevant_columns')
    print(file_path)
