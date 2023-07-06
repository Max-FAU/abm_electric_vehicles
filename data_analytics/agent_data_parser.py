import pandas as pd
import ast
import matplotlib.pyplot as plt


path2 = r'C:\Users\Max\PycharmProjects\mesa\results\model_30_025_interaction_false_norm_cars\results_run_1_agent_data.csv'
path3 = r'C:\Users\Max\PycharmProjects\mesa\results\model_30_300_interaction_false_norm_cars\results_run_1_agent_data.csv'
path = r'C:\Users\Max\PycharmProjects\mesa\results\model_30_150_interaction_true_off_peak_cars\results_run_1_agent_data.csv'
df = pd.read_csv(path)
# reduce the df to car data
df = df[df['car_data'].apply(lambda x: len(x) > 2)].reset_index(drop=True)
df = df.drop(['customer_data'], axis=1)

df[['timestamp', 'car_data']] = df['car_data'].str.split(')', expand=True)
df['car_id'] = df['timestamp'].apply(lambda x: x[11:16].replace(",", "").replace("'", ""))
df['timestamp'] = df['timestamp'].apply(lambda x: x[-20:].replace("'", ""))
# create string looking like a dict
df['car_data'] = df['car_data'].apply(lambda x: "{" + x[1:].strip())
# convert the string into a dict
df['dict_column'] = df['car_data'].apply(ast.literal_eval)
new_columns = pd.DataFrame(df['dict_column'].tolist())
df = pd.concat([df, new_columns], axis=1)
# drop original columns
df = df.drop(['car_data', 'dict_column'], axis=1)

dtypes = {
    'car_id': int,
    'recharge_value': float,
    'battery_level': float,
    'soc': float,
    'charging_priority': int,
    'plugged_in': bool,
    'battery_capacity': int,
    'trip_number': int,
    'deltapos': float,
    'cluster': int,
    'consumption': float,
    'panel_session': int,
    'terminal': int,
    'plug_in_buffer': bool,
    'target_soc_reached': bool,
    'charging_power_car': int,
    'charging_power_station': int
}

df = df.astype(dtypes)
unique_car_ids = set(df['car_id'])

def get_data_one_agent(df, id):
    df = df[df['car_id'] == id].reset_index(drop=True)
    return df

# Set first row battery lvl to 100
def calc_new_battery_level(old_battery_lvl, battery_capacity, consumption):
    if old_battery_lvl is None:
        old_battery_lvl = battery_capacity
    new_battery_lvl = old_battery_lvl - consumption
    # to not get battery lvl below 0
    new_battery_lvl = max(new_battery_lvl, 0)
    return new_battery_lvl

def calc_soc(battery_level, battery_capacity):
    soc = (battery_level / battery_capacity) * 100
    return soc

for id in unique_car_ids:
    test = get_data_one_agent(df, id)
    # filtered_test = test[(test['soc'] < 1) &
    #                      (test['consumption'] > 0)]
    # print(filtered_test['timestamp'], filtered_test['soc'], filtered_test['consumption'])
    # df_to_plot = filtered_test[['timestamp', 'soc', 'consumption', 'charging_power_car', 'charging_power_station', 'battery_level', 'recharge_value']]
    test['battery_level'] = test.apply(lambda row: calc_new_battery_level(row['battery_level'], row['battery_capacity'], row['consumption']), axis=1)
    test['soc'] = test.apply(lambda row: calc_soc(row['battery_level'], row['battery_capacity']), axis=1)
    test = test[['timestamp', 'soc', 'consumption', 'charging_power_car', 'charging_power_station', 'battery_level', 'recharge_value']]
    test.set_index('timestamp', inplace=True)
    test.plot()
    plt.show()
    break

    # ich will jetzt wissen wie viele Stunden jeder Agent mit SOC == 0 noch consumption > 0 hatte