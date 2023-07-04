import json
import numpy as np
from collections import Counter
import glob
import os
import pandas as pd

car_values = r'C:\Users\Max\PycharmProjects\mesa\input\car_values.json'
## File to count the different car models for each simulation run

def generate_cars_according_to_dist(seed_value, num_cars):
    with open(car_values, 'r') as f:
        data = json.load(f)

    total_cars = 0
    for name in data.keys():
        total_cars += data[name]["number"]

    cars = []
    distribution = []
    for name in data.keys():
        cars += [name]
        distribution += [data[name]["number"] / total_cars]

    np.random.seed(seed_value)
    car_models = np.random.choice(cars, size=num_cars, p=distribution)
    return car_models

def all_seeds():
    directory = 'C:/Users/Max/PycharmProjects/mesa/results'
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    results = {}

    for folder_name in folders:
        dicts = []
        num_cars = folder_name[9:12]
        folder_path = os.path.join(directory, folder_name)
        txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)

            with open(file_path, 'r') as file:
                number = int(file.read())
                models = generate_cars_according_to_dist(seed_value=number, num_cars=int(num_cars))
                models = Counter(models)
                dicts.append(models)

        # Calculate the sum of values for each car model
        total_counts = sum(dicts, Counter())

        # Calculate the average by dividing the total counts by the number of dictionaries
        average_counts = {car_model: count / len(dicts) for car_model, count in total_counts.items()}
        results[folder_name] = average_counts

    df = pd.DataFrame(results)
    df = df.transpose()
    df.to_csv('car_models_all_runs.csv')


if __name__ == '__main__':

    all_seeds()


