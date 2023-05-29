import dask.dataframe as dd
import os
import json
import pandas as pd

folder = f'D:\Max_Mobility_Profiles\quarterly_simulation'
files = os.listdir(folder)

relevant_files = []
for file in files:
    if file.startswith('quarterly_simulation_'):
        relevant_files.append(file)


def is_private_car(unique_id: int):
    with open(r'C:\Users\Max\PycharmProjects\mesa\input\private_cars.json') as f:
        private_cars = json.load(f)

    car_ids = []
    for entry in private_cars:
        car_ids.append(entry['id'])
    if unique_id in car_ids:
        return True
    else:
        return False

file_descriptions = {}

for file in relevant_files:
    file_path = os.path.join(folder, file)
    df = dd.read_csv(file_path)
    id_to_check = int(file.split("_")[-1].replace('.csv', ''))
    if is_private_car(unique_id=id_to_check):
        label = 'private'
    else:
        label = 'commercial'
    pd.to_datetime(df['TIMESTAMP'])
    min_timestamp = min(df['TIMESTAMP'])
    max_timestamp = max(df['TIMESTAMP'])
    average_next_timestamp = df['DELTATIME'].mean().compute()

    trip_len = df.groupby('TRIPNUMBER')['DELTAPOS'].sum().compute()
    avg_trip_len = trip_len.agg('mean')
    med_trip_len = trip_len.agg('median')

    file_descriptions[file] = {'id': id_to_check,
                               'length': len(df),
                               'cols': len(df.columns),
                               'label': label,
                               'earliest_entry': min_timestamp,
                               'latest_entry': max_timestamp,
                               'timestamp_density': average_next_timestamp,
                               'median_trip_distance': med_trip_len,
                               'average_trip_distance': avg_trip_len}

df = pd.DataFrame.from_dict(file_descriptions, orient='index')
df.to_csv('mobility_data_statistics_2.csv')


