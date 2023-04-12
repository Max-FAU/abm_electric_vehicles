import pandas as pd
import mobility_data as md
import json
import numpy as np
from car_agent import ElectricVehicle
from mobility_data import MobilityDataAggregator
import random


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


# number = number_of_each_car(car_name='renault_zoe', car_models_list=car_models)
# print(number)


if __name__ == '__main__':
    car_models = generate_cars_according_to_dist(909)
    print("Created a list containing following car models: ")
    for model in set(car_models):
        number = number_of_each_car(car_model=model, car_models_list=car_models)
        print("{} {}".format(number, model))

    # path = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    path = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    raw_mobility_data = pd.read_csv(path)

    results = {}

    i = 0
    for car in car_models:
        battery_size = random.choice(['small', 'normal', 'large'])
        individual_identifier = i
        generated_car = ElectricVehicle(model=car, battery_size=battery_size)
        generated_car.add_mobility_data(mobility_data=raw_mobility_data,
                                        starting_date='2008-07-13',
                                        num_days=1)
        # TODO fix generated_car.unique_id
        # print(generated_car.unique_id)
        timestamps = []

        for timestamp, data_row in generated_car.mobility_data.iterrows():
            battery_level = generated_car.calculate_battery_level(consumption=data_row['ECONSUMPTIONKWH'],
                                                                  battery_efficiency=100)
            timestamps.append(timestamp)

        outcome = pd.DataFrame(
            {
                'timestamp': timestamps,
                'battery_level': generated_car.battery_level_curve,
                'load_curve': generated_car.load_curve,
                'soc': generated_car.soc_curve,
                'id': generated_car.unique_id
            }
        ).set_index('timestamp')

        results[individual_identifier] = outcome
        i += 1

    # Aggregation by 15 Mins
    df_results_aggregated = pd.concat(results.values(), axis=0).groupby(pd.Grouper(freq='15Min')).mean()

    # Print the concatenated dataframe
    df_results_aggregated.to_csv('aggregated_results.csv')
