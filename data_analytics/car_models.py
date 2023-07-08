import json
import numpy as np
from collections import Counter
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


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



def create_car_models_distribution_chart():
    # Create a DataFrame from the provided data
    data = {
        'Car Model': ['BMW I3', 'FIAT 500', 'HYUNDAI KONA', 'RENAULT ZOE', 'SMART FORTWO', 'TESLA MODEL 3', 'VW GOLF',
                      'VW ID.3', 'VW ID.4, ID.5', 'VW UP'],
        'Number': [39013, 29035, 40374, 84450, 47683, 56902, 26891, 48483, 25831, 50859],
        'Distribution': ['8,7%', '6,5%', '9,0%', '18,8%', '10,6%', '12,7%', '6,0%', '10,8%', '5,7%', '11,3%']
    }

    df = pd.DataFrame(data)
    df = df.sort_values('Number', ascending=False)
    labels = df['Car Model']
    distribution = df['Distribution']
    number = df['Number']

    fig, (ax, ax_table) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3, 1]))

    ax_table.axis('off')
    ax.bar(labels, number, edgecolor="black", color='lightgrey', width=1)

    ax.set_xlim(-0.5, len(labels)-0.5)
    ax.set_ylabel('Number')
    ax.set_title('Car Model and Number')
    ax.legend(frameon=False)

    ax.tick_params(axis='x', labelrotation=90)

    table_data = pd.DataFrame({'Distribution': distribution, 'Number': number}).T
    # table_data = table_data.reset_index()
    row_labels = ['Distribution', 'Number']
    table = ax_table.table(cellText=table_data.values, rowLabels=row_labels, colLabels=None, loc='top', cellLoc='center', bbox=[0, 0.225, 1, 1])

    [t.auto_set_font_size(False) for t in [table]]
    [t.set_fontsize(8) for t in [table]]

    plt.tight_layout()
    plt.savefig("car_distribution", dpi=300)
    plt.show()



def create_charging_station_distribution_chart():
    # Create a DataFrame from the provided data
    data = {
        'Charging Station': ['0 - 3,7 kW', '> 3,7 - 15 kW', '> 15 - 22 kW', '> 22 - 49 kW', '> 49 - 59 kW',
                                        '> 59 - 149 kW', '> 149 - 299 kW', '> 299 kW'],
        'Number': [1922, 12832, 57687, 1696, 3458, 1283, 5511, 3927]
    }

    df = pd.DataFrame(data)

    labels = df['Charging Station']
    # distribution = df['Distribution']
    number = df['Number']

    fig, (ax, ax_table) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3, 1]))
    patterns = ['', '////', '////', '', '', '', '', '', '']

    ax_table.axis('off')
    for i in range(len(labels)):
        ax.bar(labels[i], number[i], edgecolor="black", color='lightgrey', hatch=patterns[i], width=1)

    ax.set_xlim(-0.5, len(labels)-0.5)
    ax.set_ylabel('Number')
    ax.set_title('Charging Station and Number')
    ax.legend(frameon=False)

    ax.tick_params(axis='x', labelrotation=90)

    table_data = pd.DataFrame({'Number': number}).T
    # table_data = table_data.reset_index()
    row_labels = ['Number']
    table = ax_table.table(cellText=table_data.values, rowLabels=row_labels, colLabels=None, loc='top', cellLoc='center', bbox=[0, 0.225, 1, 1])

    [t.auto_set_font_size(False) for t in [table]]
    [t.set_fontsize(8) for t in [table]]

    plt.tight_layout()
    plt.savefig("charging_station_distribution", dpi=300)
    plt.show()


def car_models_per_scenario():
    data = {
        'vw_id4_id5': [1.53, 1.5, 1.27, 1.1, 2.83, 2.63, 2.97, 3.33, 8.7, 8.67, 9.2, 8.03, 17.27, 18.3, 16.7, 17.63],
        'vw_id3': [3.03, 2.5, 2.53, 3.3, 5.0, 4.6, 6.1, 5.07, 15.0, 15.4, 15.87, 15.33, 32.77, 31.17, 32.27, 32.4],
        'fiat_500': [1.67, 1.8, 1.63, 1.5, 3.03, 3.1, 2.7, 3.17, 9.93, 9.83, 9.6, 9.13, 20.43, 19.53, 18.63, 20.5],
        'vw_golf': [1.8, 1.73, 1.73, 0.97, 2.77, 2.8, 3.2, 3.03, 9.3, 9.33, 9.57, 9.1, 17.57, 18.63, 18.73, 18.87],
        'vw_up': [2.77, 2.8, 2.67, 2.93, 5.73, 5.73, 5.37, 5.6, 16.13, 16.37, 16.23, 17.23, 34.43, 35.0, 35.07, 32.57],
        'renault_zoe': [4.77, 4.63, 4.6, 5.0, 9.4, 9.3, 9.03, 9.1, 29.67, 30.07, 28.3, 28.67, 55.73, 56.1, 54.07,
                        55.03],
        'smart_fortwo': [2.5, 2.53, 2.33, 2.3, 5.57, 5.83, 5.4, 5.4, 15.47, 15.67, 15.43, 15.87, 32.2, 32.9, 33.33,
                         32.83],
        'tesla_model_3': [2.97, 3.27, 3.4, 2.73, 6.17, 6.13, 6.23, 6.53, 19.3, 18.83, 19.93, 18.83, 35.87, 36.7, 37.97,
                          37.6],
        'bmw_i3': [1.7, 2.13, 2.47, 2.6, 4.83, 4.8, 4.2, 4.27, 12.9, 12.37, 11.97, 13.9, 26.17, 24.9, 26.7, 25.3],
        'hyundai_kona': [2.27, 2.1, 2.37, 2.57, 4.67, 5.07, 4.8, 4.5, 13.6, 13.47, 13.9, 13.9, 27.57, 26.77, 26.53,
                         27.27],
        # 'total': [25.0, 25.0, 25.0, 25.0, 50.0, 50.0, 50.0, 50.0, 150.0, 150.0, 150.0, 150.0, 300.0, 300.0, 300.0,
        #           300.0],
        'name': [
    '025_interaction_false_norm_cars',
    '025_interaction_false_off_peak_cars',
    '025_interaction_true_norm_cars',
    '025_interaction_true_off_peak_cars',
    '050_interaction_true_off_peak_cars',
    '050_interaction_false_norm_cars',
    '050_interaction_false_off_peak_cars',
    '050_interaction_true_norm_cars',
    '150_interaction_false_norm_cars',
    '150_interaction_false_off_peak_cars',
    '150_interaction_true_norm_cars',
    '150_interaction_true_off_peak_cars',
    '300_interaction_false_norm_cars',
    '300_interaction_false_off_peak_cars',
    '300_interaction_true_norm_cars',
    '300_interaction_true_off_peak_cars'
]
    }

    name_mapping = {
        '025_interaction_false_norm_cars': 'Scenario 01',
        '025_interaction_true_norm_cars': 'Scenario 02',
        '025_interaction_false_off_peak_cars': 'Scenario 03',
        '025_interaction_true_off_peak_cars': 'Scenario 04',
        '050_interaction_false_norm_cars': 'Scenario 05',
        '050_interaction_true_norm_cars': 'Scenario 06',
        '050_interaction_false_off_peak_cars': 'Scenario 07',
        '050_interaction_true_off_peak_cars': 'Scenario 08',
        '150_interaction_false_norm_cars': 'Scenario 09',
        '150_interaction_true_norm_cars': 'Scenario 10',
        '150_interaction_false_off_peak_cars': 'Scenario 11',
        '150_interaction_true_off_peak_cars': 'Scenario 12',
        '300_interaction_false_norm_cars': 'Scenario 13',
        '300_interaction_true_norm_cars': 'Scenario 14',
        '300_interaction_false_off_peak_cars': 'Scenario 15',
        '300_interaction_true_off_peak_cars': 'Scenario 16'
    }



    df = pd.DataFrame(data)
    df['name'] = df['name'].apply(lambda x: name_mapping.get(x))
    df = df.sort_values(by='name')

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        data = df.iloc[i, :-1]  # Exclude the last column 'name'
        data = data.sort_values(ascending=False)
        index_name = df.iloc[i, -1]  # Last column 'name' as the index name
        bars = ax.bar(data.index, data.values, color='grey', edgecolor='black')

        ax.set_title(index_name)
        ax.xaxis.set_visible(False)  # Hide x-axis labels for all subplots

        # Show y-axis tick labels for the first column of subplots
        if i % 4 == 0:
            ax.yaxis.set_visible(True)
            ax.set_ylabel('Number')

        # Show x-axis tick labels for the last row of subplots
        if i >= len(axes) - 4:
            ax.xaxis.set_visible(True)
            ax.tick_params(axis='x', rotation=90)

        # Add value annotations to each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 2), ha='center', va='top', rotation=90)

    fig.tight_layout()
    plt.savefig('model_overview', dpi=300)
    plt.show()



if __name__ == '__main__':

    # all_seeds()
    create_car_models_distribution_chart()
    # create_charging_station_distribution_chart()
    # car_models_per_scenario()

