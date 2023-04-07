import pandas as pd
import mobility_data as md
import json
import numpy as np


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

    car_names = np.random.choice(cars, size=number_of_agents, p=distribution)
    # print(len(car_names), "car names generated.")

    return car_names


def number_of_each_car(car_name, car_names_list):
    car_series = pd.Series(car_names_list)
    car_counts = car_series.value_counts()
    if car_name in car_counts.index:
        return car_counts[car_name]
    else:
        return 0

# car_models = generate_cars_according_to_dist(10)
# number = number_of_each_car(car_name='renault_zoe', car_names_list=car_models)
# print(number)



if __name__ == '__main__':
    path = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    mobility_data = pd.read_csv(path)
    mobility_data = md.prepare_mobility_data(df=mobility_data,
                                          starting_date='2008-07-12 00:00:00',
                                          days=1)

    mobility_data_aggregated = md.aggregate_15_min_steps(mobility_data)
    print(mobility_data_aggregated)