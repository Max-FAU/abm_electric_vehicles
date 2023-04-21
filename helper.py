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


if __name__ == '__main__':
    file_path = read_json_config('relevant_columns')
    print(file_path)
