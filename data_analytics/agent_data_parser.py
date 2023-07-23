import pandas as pd
import ast
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import itertools

import auxiliary as aux


def get_agent_files(interaction: bool,
                    car_type: str,
                    number_cars: int,
                    test=False):
    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    if number_cars == 25:
        number_cars = '025'
    elif number_cars == 50:
        number_cars = '050'

    number_cars_str = str(number_cars)

    if interaction:
        interaction_str = '_true_'
    elif interaction is False:
        interaction_str = '_false_'
    else:
        interaction_str = '_*_'

    if car_type == 'normal':
        car_type_str = 'norm_cars'
    elif car_type == 'offpeak':
        car_type_str = 'off_peak_cars'
    else:
        car_type_str = '*'
    total_string = 'model_30_' + number_cars_str + '_interaction' + interaction_str + car_type_str

    if test:
        files = "/*1_agent_data.csv"
    else:
        files = "/*_agent_data.csv"
    end_string = total_string + files
    csv_files = list(directory.glob(end_string))
    csv_files = sorted(csv_files)
    return csv_files


def parse_agent_data(agent_data: pd.DataFrame):
    # reduce the df to car data
    df = agent_data[agent_data['car_data'].apply(lambda x: len(x) > 2)].reset_index(drop=True)
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
    return df


def get_data_one_agent(df, id):
    df = df[df['car_id'] == id].reset_index(drop=True)
    return df

    # # Set first row battery lvl to 100
    # def calc_new_battery_level(old_battery_lvl, battery_capacity, consumption):
    #     if old_battery_lvl is None:
    #         old_battery_lvl = battery_capacity
    #     new_battery_lvl = old_battery_lvl - consumption
    #     # to not get battery lvl below 0
    #     new_battery_lvl = max(new_battery_lvl, 0)
    #     return new_battery_lvl
    #
    # def calc_soc(battery_level, battery_capacity):
    #     soc = (battery_level / battery_capacity) * 100
    #     return soc

def driving_no_capacity(df):
    total_results = []
    unique_car_ids = set(df['car_id'])
    for id in unique_car_ids:
        test = get_data_one_agent(df, id)
        test = test[['timestamp', 'charging_priority', 'soc', 'consumption', 'charging_power_car', 'charging_power_station', 'battery_level', 'recharge_value']]
        test.set_index('timestamp', inplace=True)
        total_timestamps = len(test)

        driving_no_capacity = test[(test['soc'] <= 0) & (test['consumption'] > 0)]
        timestamps_no_capa = len(driving_no_capacity)
        driving = len(test[test['consumption'] > 0])
        if timestamps_no_capa == 0:
            percentage = 0
        else:
            percentage = timestamps_no_capa / driving * 100
        average_prio = test['charging_priority'].mean()

        color = 'tab:green'
        name = str(csv_file)
        if 'false' in name:
            color = 'tab:red'
        if 'true' in name:
            color = 'tab:blue'

        # print("id {}, "
        #       "total_timestamps {}, "
        #       "driving {}, "
        #       "timestamps_no_capa {}, "
        #       "percentage {}, "
        #       "average_prio {}".format(id, total_timestamps, driving, timestamps_no_capa, round(percentage, 2), round(average_prio, 2)))

        result = (id, total_timestamps, driving, timestamps_no_capa, round(percentage, 2), round(average_prio, 2), color)
        total_results.append(result)

    result_df = pd.DataFrame(total_results,
                             columns=['id', 'total_timestamps', 'driving', 'timestamps_no_capa', 'percentage',
                                      'average priority', 'color'])
    return result_df


def create_plot_driving_no_capa(result_df,
                                scatter=False,
                                violin=False):
    # result_df = pd.DataFrame(total_results, columns=['id', 'total_timestamps', 'driving', 'timestamps_no_capa', 'percentage', 'average priority', 'color'])
    #     # print(result_df)
    if scatter:
        plt.scatter(result_df['average priority'], result_df['percentage'], c=result_df['color'])
        grouped_data = result_df.groupby('color')

        for color, group in grouped_data:
            label = 'interaction' if color == 'tab:blue' else 'no_interaction'
            sns.regplot(x=group['average priority'], y=group['percentage'], line_kws={'color': color}, scatter=False, scatter_kws=None, ci=None, label=label)

        plt.xlim(None, result_df['average priority'].max())
        plt.ylim(0, None)
        plt.legend()
        plt.show()

    if violin:
        result_df['interaction'] = result_df['color'].replace('tab:blue', 'interaction').replace('tab:red', 'no interaction')
        sns.violinplot(x=result_df['interaction'], y=result_df['average priority'], hue=result_df['interaction'])
        # sns.violinplot(data=group['percentage'])
        plt.show()


def spitzenlast_pro_auto():
    # agents = [25, 50, 150, 300]
    #
    # car_num = agents[2]

    scenarios = [
        "*_025_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_025_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_025_interaction_false_off*/results_run_*_agent*.csv",
        # "*_025_interaction_true_off*/results_run_*_agent*.csv",

        "*_050_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_050_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_050_interaction_false_off*/results_run_*_agent*.csv",
        # "*_050_interaction_true_off*/results_run_*_agent*.csv",

        "*_150_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_150_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_150_interaction_false_off*/results_run_*_agent*.csv",
        # "*_150_interaction_true_off*/results_run_*_agent*.csv",

        "*_300_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_300_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_300_interaction_false_off*/results_run_*_agent*.csv",
        # "*_300_interaction_true_off*/results_run_*_agent*.csv"
    ]


    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    fig, ax = plt.subplots(figsize=(12, 6))

    for j, scenario in enumerate(scenarios):

        csv_files = list(directory.glob(scenario))
        name = str(scenario)

        # if '25' in name:
        #     car_num = 25
        # if '50' in name:
        #     car_num = 50
        # if '150' in name:
        #     car_num = 150
        # if '300' in name:
        #     car_num = 300

        print('For selected scenarios there are {} files'.format(len(csv_files)))

        max_values = []

        for i, csv_file in enumerate(csv_files):
            recharge_power_list = []

            df = pd.read_csv(csv_file)
            df_parsed = parse_agent_data(agent_data=df)
            car_num = int(len(df_parsed) / 1344)
            print(car_num)
            car_ids_list = list(range(1, car_num + 1))
            repeated_car_ids = list(itertools.islice(itertools.cycle(car_ids_list), 1344 * car_num))
            df_parsed['car_id'] = repeated_car_ids
            df_parsed.set_index('timestamp', inplace=True)
            unique_ids = set(df_parsed['car_id'])
            df_parsed['recharge_power'] = df_parsed['recharge_value'].apply(
                lambda x: aux.convert_kw_kwh(kwh=x) / 0.9)

            for car_id in unique_ids:
                df_car_agent = df_parsed[df_parsed['car_id'] == car_id]
                col_name = 'recharge_power_{}_{}'.format(i, car_id)
                df_car_agent = df_car_agent.rename(columns={'recharge_power': col_name})
                if len(df_car_agent) == 1344:
                    recharge_power_list.append(df_car_agent[col_name])
                else:
                    print("")

            recharge_power_list_df = pd.concat(recharge_power_list, axis=1)
            # print(recharge_power_list_df.shape)

            recharge_power_list_df['total'] = recharge_power_list_df.sum(axis=1)
            recharge_power_list_df['total_per_car'] = recharge_power_list_df['total'] / car_num
            print(recharge_power_list_df['total_per_car'].max())
            max_values.append([recharge_power_list_df['total_per_car'].max()])

        plt.scatter([j + 1] * len(max_values), max_values, color='blue')

    custom_xticks_positions = [0, 1, 2, 3, 4, 5]
    custom_xticklabels = ['', '25 EVs', '50 EVs', '150 EVs', '300 EVs', '']
    plt.title('Peak Load Charging: Different Fleet Sizes')
    plt.ylabel('Peak Load per EV\n[kW]')
    plt.xlabel('Fleet Size')
    plt.xticks(custom_xticks_positions, custom_xticklabels, rotation=90)
    plt.tight_layout()
    plt.savefig('Coincidence Effect Charging.png', dpi=100)
    plt.show()


def plot_agent_min_mean_max_recharge_power():
    num_cars = 300
    # 25, 50, 150, 300
    capacity = [33.75, 67.5, 225.0, 299.7]
    capacity = capacity[3]

    # normal, offpeak
    files = get_agent_files(interaction=False,
                            car_type='normal',
                            number_cars=num_cars,
                            test=False)

    all_files = files
    data = []

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, csv_file in enumerate(all_files):
        df = pd.read_csv(csv_file)
        df_parsed = parse_agent_data(agent_data=df)
        df_parsed['recharge_power'] = df_parsed['recharge_value'].apply(lambda x: aux.convert_kw_kwh(kwh=x) / 0.9)
        df_parsed['timestamp'] = pd.to_datetime(df_parsed['timestamp'])
        df_parsed.set_index('timestamp', inplace=True)
        # Sum the recharging power values for all agents in one agent data file
        entry = df_parsed.resample('15T')['recharge_power'].sum()
        data.append(entry)

    data_df = pd.concat(data, axis=1)

    data_df['min_recharge'] = data_df.apply(lambda row: row.min(), axis=1)
    data_df['mean_recharge'] = data_df.apply(lambda row: row.mean(), axis=1)
    data_df['max_recharge'] = data_df.apply(lambda row: row.max(), axis=1)

    # plt.plot(data_df.index, data_df['min_recharge'], label='Min Power')
    # plt.plot(data_df.index, data_df['mean_recharge'], label='Avg Power')
    plt.plot(data_df.index, data_df['max_recharge'], label='Max Power')

    # plt.plot(data_df.index, [capacity] * len(data_df), label='Transformer Capacity', color='red')
    anzahl_viertel_stunden = len(data_df[data_df['max_recharge'] > capacity])
    print(anzahl_viertel_stunden)
    #
    # plt.xlim(data_df.index.min(), data_df.index.max())
    # plt.xticks(rotation=90)
    # plt.ylabel('Recharge Power \n [kW]')
    # plt.title('Maximum Recharging Power')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()
    # plt.show()


def plot_one_agent_charging_load():

    agents = [25, 50, 150, 300]

    scenarios = [
        "*_025_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_025_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_025_interaction_false_off*/results_run_*_agent*.csv",
        # "*_025_interaction_true_off*/results_run_*_agent*.csv",

        # "*_050_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_050_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_050_interaction_false_off*/results_run_*_agent*.csv",
        # "*_050_interaction_true_off*/results_run_*_agent*.csv",

        # "*_150_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_150_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_150_interaction_false_off*/results_run_*_agent*.csv",
        # "*_150_interaction_true_off*/results_run_*_agent*.csv",

        # "*_300_interaction_false_norm*/results_run_*_agent*.csv",
        # "*_300_interaction_true_norm*/results_run_*_agent*.csv",
        # "*_300_interaction_false_off*/results_run_*_agent*.csv",
        # "*_300_interaction_true_off*/results_run_*_agent*.csv"
    ]


    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    fig, ax = plt.subplots(figsize=(12, 6))

    recharge_power_list = []

    for j, scenario in enumerate(scenarios):
        csv_files = list(directory.glob(scenario))
        print('For selected scenarios there are {} files'.format(len(csv_files)))

        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            df_parsed = parse_agent_data(agent_data=df)
            df_parsed.set_index('timestamp', inplace=True)
            unique_ids = set(df_parsed['car_id'])
            df_parsed['recharge_power'] = df_parsed['recharge_value'].apply(
                lambda x: aux.convert_kw_kwh(kwh=x) / 0.9)

            for car_id in unique_ids:
                df_car_agent = df_parsed[df_parsed['car_id'] == car_id]
                col_name = 'recharge_power_{}_{}'.format(i, car_id)
                df_car_agent = df_car_agent.rename(columns={'recharge_power': col_name})
                if len(df_car_agent) == 1344:
                    recharge_power_list.append(df_car_agent[col_name])
                else:
                    print(len(df_car_agent), ' is not equal to 1344.')

    recharge_power_list_df = pd.concat(recharge_power_list, axis=1)
    # print(recharge_power_list_df.shape)

    recharge_power_list_df['average'] = recharge_power_list_df.mean(axis=1)
    recharge_power_list_df['percentile_5'] = recharge_power_list_df.quantile(0.05, axis=1)
    recharge_power_list_df['percentile_95'] = recharge_power_list_df.quantile(0.95, axis=1)

    # recharge_power_list_df['capacity'] = [capacity[0] / agents[0]] * len(recharge_power_list_df)

    recharge_power_list_df.plot(y='average', ax=ax, label='Average Charging Profile', color='blue', linestyle='dotted', linewidth=1.5)
    # recharge_power_list_df.plot(y='capacity', ax=ax, label='Transformer Capacity 100%', color='green', linewidth=1.5)

    recharge_power_list_df.plot(y='percentile_5', ax=ax, label='5% Percentile', color='darkgrey', linewidth=0.5)
    recharge_power_list_df.plot(y='percentile_95', ax=ax, label='95% Percentile', color='darkgrey', linewidth=0.5)

    ax.fill_between(recharge_power_list_df.index, recharge_power_list_df['percentile_5'], recharge_power_list_df['percentile_95'], color='grey',
                    alpha=0.1, label='5% - 95% Percentile')

    plt.title('Agent Data\nCharging Profile - Average and Percentiles')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()

    plt.show()


def calculate_agent_statistics():
    j = 13  # TODO replace this
    num_cars_list = [300]   # TODO Only for 300 run [25, 50, 150]
    car_types_list = ['normal', 'offpeak']
    interaction_list = [True, False]
    # 25, 50, 150, 300
    capacity = [33.75, 67.5, 225.0, 299.7]
    capacity = capacity[1]

    for num_cars in num_cars_list:
        for car_type in car_types_list:
            for interaction in interaction_list:
                print(num_cars, car_type, interaction)
                files = get_agent_files(interaction=interaction,
                                        car_type=car_type,
                                        number_cars=num_cars,
                                        test=False)

                all_files = files
                data = []

                fig, ax = plt.subplots(figsize=(12, 6))

                for i, csv_file in enumerate(all_files):
                    df = pd.read_csv(csv_file)
                    df_parsed = parse_agent_data(agent_data=df)
                    df_parsed.set_index('timestamp', inplace=True)
                    total_timestamps = len(df_parsed)

                    driving_no_capacity = df_parsed[(df_parsed['soc'] <= 0) & (df_parsed['consumption'] > 0)]
                    timestamps_no_capa = len(driving_no_capacity)

                    driving = len(df_parsed[df_parsed['consumption'] > 0])
                    if timestamps_no_capa == 0:
                        percentage = 0
                    else:
                        percentage = timestamps_no_capa / driving * 100
                    average_prio = df_parsed['charging_priority'].mean()
                    median_prio = df_parsed['charging_priority'].median()

                    color = 'tab:green'
                    name = str(csv_file)
                    if 'false' in name:
                        color = 'tab:red'
                    if 'true' in name:
                        color = 'tab:blue'

                        # print("id {}, "
                        #       "total_timestamps {}, "
                        #       "driving {}, "
                        #       "timestamps_no_capa {}, "
                        #       "percentage {}, "
                        #       "average_prio {}".format(id, total_timestamps, driving, timestamps_no_capa, round(percentage, 2), round(average_prio, 2)))

                    average_soc = df_parsed['soc'].mean()
                    median_soc = df_parsed['soc'].median()
                    recharge_sum = df_parsed['recharge_value'].sum() # sum
                    consumption_sum = df_parsed['consumption'].sum() # sum
                    time_plugged_in = df_parsed['plugged_in'].sum()  # length / number of plugged in quarter hours

                    result = (
                        i,
                        total_timestamps,
                        driving,
                        timestamps_no_capa,
                        round(percentage, 2),
                        round(average_prio, 2),
                        round(median_prio, 2),
                        round(average_soc, 2),
                        round(median_soc, 2),
                        recharge_sum,
                        consumption_sum,
                        time_plugged_in,
                        color)

                    data.append(result)

                result_df = pd.DataFrame(data,
                                         columns=['id',
                                                  'total_timestamps',
                                                  'driving',
                                                  'timestamps_no_capa',
                                                  'percentage',
                                                  'average priority',
                                                  'median priority',
                                                  'average soc',
                                                  'median soc',
                                                  'recharge sum',
                                                  'consumption sum',
                                                  'time plugged in',
                                                  'color'])

                result_df.to_csv('results_{}.csv'.format(j), index=True)
                j += 1


def read_agent_statistics_results():
    directory_path = r'C:\Users\Max\PycharmProjects\mesa\data_analytics'
    directory = Path(directory_path)
    file = 'results_*.csv'
    csv_files = list(directory.glob(file))
    csv_files = sorted(csv_files)

    fig, ax = plt.subplots()

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        # df = df.sort_values(by='percentage', ascending=False)
        # df.reset_index(inplace=True)
        print(df.columns)
        try:
            color = df['color'].iloc[1]
            ax.plot(df['driving'], color=color)
        except:
            continue
    plt.show()


def charging_priority_one_agent_example():
    df = pd.read_csv(r"C:\Users\Max\Desktop\example_charging_prio.csv", header=None)
    df.columns = ['timestamp', 'soc', 'soc_prio', 'next_trip_consumption', 'next_trip_prio', 'charging_duration', 'prio_time']
    df['timestamp'] = df['timestamp'].str.replace('timestamp', '')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    df['soc'] = df['soc'].str.replace('soc', '').astype(float)
    df['soc_prio'] = df['soc_prio'].str.replace('prio_soc', '').astype(int)
    df['next_trip_consumption'] = df['next_trip_consumption'].str.replace('next_trip', '').astype(float)
    df['next_trip_prio'] = df['next_trip_prio'].str.replace('prio_next_trip', '').astype(int)
    df['charging_duration'] = df['charging_duration'].str.replace('charging_duration', '').astype(float)
    df['prio_time'] = df['prio_time'].str.replace('prio_time', '').astype(int)

    df['charging_prio'] = df['soc_prio'] + df['next_trip_prio'] + df['prio_time']

    # df.plot()
    # plt.show()
    # print(df)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # ax.plot(df_parsed)
    # ax1.plot(df.index, df['soc'], color='blue')
    ax2 = ax1.twinx()
    # Plotting on ax2 and filling areas
    ax2.plot(df.index, df['soc_prio'], color='green', label='SOC Priority')
    ax2.fill_between(df.index, df['soc_prio'], color='green', alpha=0.2, hatch='//////')

    ax2.plot(df.index, df['soc_prio'] + df['next_trip_prio'], color='orange', label='Consumption Priority')
    ax2.fill_between(df.index, df['soc_prio'], df['soc_prio'] + df['next_trip_prio'], color='orange', alpha=0.2, hatch='//////')

    ax2.plot(df.index, df['soc_prio'] + df['next_trip_prio'] + df['prio_time'], color='red', label='Time Priority')
    ax2.fill_between(df.index, df['soc_prio'] + df['next_trip_prio'],
                     df['soc_prio'] + df['next_trip_prio'] + df['prio_time'], color='red', alpha=0.2, hatch='//////')

    # To plot 'ax1' on top of the filled areas, you can simply call 'ax1' plot again after plotting on 'ax2'
    ax1.plot(df.index, df['soc'], linewidth=2, color='blue', label='SOC')
    # ax2.plot(df.index, df['charging_prio'], color='red')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax2.legend(loc='lower left', bbox_to_anchor=(1.05, 0.8), borderaxespad=0.)

    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Set x-axis limits to start from 0 and end at the maximum timestamp
    # ax1.set_xlim(0, None)

    plt.xticks(rotation=90)
    plt.title('Example Charging Priority')
    ax1.set_ylabel('SOC\n [%]')
    ax2.set_ylabel('Priorities')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # plot_agent_min_mean_max_recharge_power()
    # calculate_agent_statistics()  # TODO run for 300
    # read_agent_statistics_results()  # TODO implement good statistics to show that car has enough time to charge
    # charging_priority_one_agent_example()   # create one example plot for charging priority
    # plot_one_agent_charging_load()
    spitzenlast_pro_auto()