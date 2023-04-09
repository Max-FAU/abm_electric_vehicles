import json


def read_json_config(keyword):
    with open('relevant_columns_config.json', 'r') as config:
        columns = json.load(config)
    return columns[keyword]


if __name__ == '__main__':
    file_path = read_json_config('relevant_columns')
    print(file_path)
