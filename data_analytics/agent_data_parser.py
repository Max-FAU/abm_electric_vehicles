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
    j = 16  # TODO replace this
    num_cars_list = [300]   # TODO Only for 300 run [25, 50, 150]
    # car_types_list = ['normal', 'offpeak']
    car_types_list = ['offpeak']
    # interaction_list = [True, False]
    interaction_list = [True]
    # 25, 50, 150, 300
    capacity = [33.75, 67.5, 225.0, 299.7]
    capacity = capacity[3]

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
    breakpoint()


def read_agent_statistics_results():
    # files = ['results_1.csv', 'results_2.csv', 'results_3.csv', 'results_4.csv', 'results_5.csv',
    #          'results_6.csv', 'results_7.csv', 'results_8.csv', 'results_9.csv', 'results_10.csv',
    #          'results_11.csv', 'results_12.csv', 'results_13.csv', 'results_14.csv', 'results_15.csv',
    #          'results_16.csv']
    #
    # title = ['Scenario 01', 'Scenario 02', 'Scenario 03', 'Scenario 04', 'Scenario 05',
    #           'Scenario 06', 'Scenario 07', 'Scenario 08', 'Scenario 09', 'Scenario 10',
    #           'Scenario 11', 'Scenario 12', 'Scenario 13', 'Scenario 14', 'Scenario 15',
    #           'Scenario 16']

    # color = ['blue', 'green', 'grey', 'red'] * 4
    color = ['blue', 'green', 'grey', 'red']
    directory_path = r'C:\Users\Max\PycharmProjects\mesa\data_analytics'
    directory = Path(directory_path)
    files = ['results_13.csv', 'results_14.csv', 'results_15.csv', 'results_16.csv']
    title = ['Scenario 13', 'Scenario 14', 'Scenario 15', 'Scenario 16']

    average_percentages = []  # percentage driving without capacity
    average_drivings = []   # timestamps driving
    average_timestamps_no_capa = []  # timestamps no capacity
    sum_recharges = []
    sum_times = []
    average_priorities = []
    average_socs = []
    consumptions = []

    for i, file in enumerate(files):
        csv_files = list(directory.glob(file))
        df = pd.read_csv(csv_files[0])

        average_percentage = df['percentage'].sum() / 30
        average_priority = df['average priority'].sum() / 30
        average_soc = df['average soc'].sum() / 30

        if 'results_1.csv' in str(file) or 'results_2.csv' in str(file) or 'results_3' in str(file) or 'results_4' in str(file):
            divisor = 25
        if 'results_5' in str(file) or 'results_6' in str(file) or 'results_7' in str(file) or 'results_8' in str(file):
            divisor = 50
        if 'results_9' in str(file) or 'results_10' in str(file) or 'results_11' in str(file) or 'results_12' in str(file):
            divisor = 150
        if '13' in str(file) or '14' in str(file) or '15' in str(file) or '16' in str(file):
            divisor = 300

        print(str(file))
        print(divisor)
        average_driving = df['driving'].sum() / 30 / divisor
        average_no_capa = df['timestamps_no_capa'].sum() / 30 / divisor
        sum_recharge_power = df['recharge sum'].sum() / 30 / divisor
        time_plugged_in = df['time plugged in'].sum() / 30 / divisor
        consumption_sum = df['consumption sum'].sum() / 30 / divisor

        # ax.plot(sorted(df['timestamps_no_capa']), label=title[i])
        average_percentages.append(average_percentage)
        average_drivings.append(average_driving)
        average_timestamps_no_capa.append(average_no_capa)
        sum_recharges.append(sum_recharge_power)
        sum_times.append(time_plugged_in)
        average_priorities.append(average_priority)
        average_socs.append(average_soc)
        consumptions.append(consumption_sum)

    # units = ['count', '%', 'count', 'kWh', 'count', 'int', '%', 'kWh']

    df = pd.DataFrame({
        # 'Units': units,
        'Timestamps Driving': average_drivings,
        # 'Timestamps 0% SOC': average_timestamps_no_capa,
        'Percentage Driving 0% SOC': average_percentages,
        'Recharged Energy': sum_recharges,
        'Timestamps Plugged In': sum_times,
        'Average Priority': average_priorities,
        'Average SOC': average_socs,
        'Consumption': consumptions
    }, index=title).round(2)

    def add_trailing_zero(val):
        try:
            return '{:.2f}'.format(float(val))
        except ValueError:
            return val

    # Apply the function to each cell of the DataFrame
    df = df.applymap(add_trailing_zero)

    new_row_data = {
        'Timestamps Driving': '[count]',
        # 'Timestamps 0% SOC': '[count]',
        'Percentage Driving 0% SOC': '[%]',
        'Recharged Energy': '[kWh]',
        'Timestamps Plugged In': '[count]',
        'Average Priority': '[number]',
        'Average SOC': '[%]',
        'Consumption': '[kWh]',
    }

    # Convert the new row data to a DataFrame
    new_row_df = pd.DataFrame(new_row_data, index=['unit'])

    # Concatenate the new row DataFrame with the original DataFrame using pd.concat
    df = pd.concat([new_row_df, df])


    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.axis('off')  # Turn off axis
    # Use the pandas DataFrame.plot() function with kind='table' to plot the table
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                     cellLoc='center', loc='center')

    # Set font size for the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_height(0.3)
            cell.set_text_props(rotation=90)

    plt.tight_layout()
    # plt.savefig('table_agent_results.png', dpi=300)
    plt.show()

    #
    # bars = ax.bar(title, averages, color=color, alpha=0.3)
    # ##'id', 'total_timestamps', 'driving', 'timestamps_no_capa',
    #    # 'percentage', 'average priority', 'median priority', 'average soc',
    #    # 'median soc', 'recharge sum', 'consumption sum', 'time plugged in',
    #    # 'color']
    # ##
    #
    # for bar, average in zip(bars, averages):
    #     height = bar.get_height()
    #     ax.annotate(f'{average:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, -20), textcoords="offset points",
    #                 ha='center', va='bottom')
    #
    # plt.title('Average Percentage of Timestamps driving with 0% SOC')
    # plt.xticks(rotation=90)
    # plt.ylabel('Percentage [%]')
    # plt.tight_layout()
    # plt.show()

def read_agent_statistics_recharge_sum():
    # files = ['results_1.csv', 'results_2.csv', 'results_3.csv', 'results_4.csv', 'results_5.csv',
    #          'results_6.csv', 'results_7.csv', 'results_8.csv', 'results_9.csv', 'results_10.csv',
    #          'results_11.csv', 'results_12.csv', 'results_13.csv', 'results_14.csv', 'results_15.csv',
    #          'results_16.csv']
    #
    # title = ['Scenario 01', 'Scenario 02', 'Scenario 03', 'Scenario 04', 'Scenario 05',
    #           'Scenario 06', 'Scenario 07', 'Scenario 08', 'Scenario 09', 'Scenario 10',
    #           'Scenario 11', 'Scenario 12', 'Scenario 13', 'Scenario 14', 'Scenario 15',
    #           'Scenario 16']

    # color = ['blue', 'green', 'grey', 'red'] * 4
    color = ['blue', 'green', 'grey', 'red']
    directory_path = r'C:\Users\Max\PycharmProjects\mesa\data_analytics'
    directory = Path(directory_path)
    files = ['results_13.csv', 'results_14.csv', 'results_15.csv', 'results_16.csv']
    title = ['Scenario 13', 'Scenario 14', 'Scenario 15', 'Scenario 16']
    # file = 'results_*.csv'

    averages = []

    fig, ax = plt.subplots()
    for i, file in enumerate(files):
        csv_files = list(directory.glob(file))
        df = pd.read_csv(csv_files[0])
        print(df.columns)
        # df = df.sort_values(by='percentage', ascending=False)
        # df.reset_index(inplace=True)
        # x_positions = range(len(df))
        average = df['recharge sum'].sum() / 300 / 30
        # ax.plot(sorted(df['timestamps_no_capa']), label=title[i])
        averages.append(average)

    bars = ax.bar(title, averages, color=color, alpha=0.3)
    ##'id', 'total_timestamps', 'driving', 'timestamps_no_capa',
       # 'percentage', 'average priority', 'median priority', 'average soc',
       # 'median soc', 'recharge sum', 'consumption sum', 'time plugged in',
       # 'color']
    ##

    for bar, average in zip(bars, averages):
        height = bar.get_height()
        ax.annotate(f'{average:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -20), textcoords="offset points",
                    ha='center', va='bottom')

    plt.title('Sum of Charged Energy per Agent')
    plt.xticks(rotation=90)
    plt.ylabel('Charged Energy [kWh]')
    plt.tight_layout()
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



def dump():
    j = 13  # TODO replace this
    num_cars_list = [300]   # TODO Only for 300 run [25, 50, 150]
    # car_types_list = ['normal', 'offpeak']
    car_types_list = ['normal']
    # interaction_list = [True, False]
    interaction_list = [False]
    # 25, 50, 150, 300
    capacity = [33.75, 67.5, 225.0, 299.7]
    capacity = capacity[3]

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
                    df_parsed.to_csv('test_agent_data_parsed_no_interaction_normal.csv')
                    print(df_parsed)
                    breakpoint()


def create_agent_plot():
    file = r"C:\Users\Max\Desktop\Master Thesis\Writing\agent_comparison.xlsx"
    dataframes_dict = pd.read_excel(file, sheet_name=None)
    small_interaction = dataframes_dict["1921_interaction"]
    small_no_interaction = dataframes_dict["1921_no_interaction"]

    large_interaction = dataframes_dict["220_interaction"]
    large_no_interaction = dataframes_dict["220_no_interaction"]

    list = [small_interaction, small_no_interaction, large_interaction, large_no_interaction]
    for df in list:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)


    fig, axs = plt.subplots(2, 1, figsize=(16, 8))

    # ax1 = axs[0]
    # ax2 = ax1.twinx()
    #
    # ax2.plot(small_no_interaction['recharge_value'], color='blue', linestyle='dotted', linewidth=2, label='Charging Value')
    # ax1.plot(small_no_interaction['soc'], color='black', label='SOC')
    # ax1.fill_between(small_no_interaction['driving'].index, 0, 120,
    #                  where=(small_no_interaction['driving'] != 0), color='red', alpha=0.3, label='Driving')
    #
    # ax1.set_title("Small Agent - No Interaction")
    # ax1.set_ylabel("SOC [%]")
    # ax2.set_ylabel("Charging Value [kWh]")
    # ax1.set_xlim(small_no_interaction.index.min(), small_no_interaction.index.max())
    # ax1.set_ylim(0, 110)
    # ax2.set_ylim(0, 1)
    # ax1.tick_params(axis='x', rotation=90)
    #
    #
    # ax3 = axs[1]
    # ax4 = ax3.twinx()
    #
    # ax4.plot(small_interaction['recharge_value'], color='blue', linestyle='dotted', linewidth=2)
    # ax3.plot(small_interaction['soc'], color='black')
    # ax3.fill_between(small_interaction['driving'].index, 0, 120,
    #                  where=(small_interaction['driving'] != 0), color='red', alpha=0.3)
    #
    # ax3.set_title("Small Agent - Interaction")
    # ax3.set_ylabel("SOC [%]")
    # ax4.set_ylabel("Charging Value [kWh]")
    # ax3.set_xlim(small_interaction.index.min(), small_interaction.index.max())
    # ax3.set_ylim(0, 110)
    # ax4.set_ylim(0, 1)
    # ax3.tick_params(axis='x', rotation=90)


    ax5 = axs[0]
    ax6 = ax5.twinx()

    ax6.plot(large_no_interaction['recharge_value'], color='blue', linestyle='dotted', linewidth=2, label='Charging Value')
    ax5.plot(large_no_interaction['soc'], color='black', label='SOC')
    ax5.fill_between(large_no_interaction['driving'].index, 0, 120,
                     where=(large_no_interaction['driving'] != 0), color='red', alpha=0.3, label='Driving')

    ax5.set_title("Large Agent - No Interaction")
    ax5.set_ylabel("SOC [%]")
    ax6.set_ylabel("Charging Value [kWh]")
    ax5.set_xlim(large_no_interaction.index.min(), large_no_interaction.index.max())
    ax5.set_ylim(0, 110)
    ax6.set_ylim(0, 2.5)
    ax5.tick_params(axis='x', rotation=90)

    ax7 = axs[1]
    ax8 = ax7.twinx()

    ax8.plot(large_interaction['recharge_value'], color='blue', linestyle='dotted', linewidth=2)
    ax7.plot(large_interaction['soc'], color='black')
    ax7.fill_between(large_interaction['driving'].index, 0, 120,
                     where=(large_interaction['driving'] != 0), color='red', alpha=0.3)

    ax7.set_title("Large Agent - Interaction")
    ax7.set_ylabel("SOC [%]")
    ax8.set_ylabel("Charging Value [kWh]")
    ax7.set_xlim(large_interaction.index.min(), large_interaction.index.max())
    ax7.set_ylim(0, 110)
    ax8.set_ylim(0, 2.5)
    ax7.tick_params(axis='x', rotation=90)

    ax5.legend(loc='upper right', bbox_to_anchor=(1.16, 1), ncol=1, frameon=False)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 0.84), ncol=1, frameon=False)

    plt.tight_layout()

    plt.show()

def plugged_in_charging_agents():
    file1 = r"C:\Users\Max\PycharmProjects\mesa\data_analytics\number_charging_agents_offpeak.xlsx"
    file2 = r"C:\Users\Max\PycharmProjects\mesa\data_analytics\number_charging_agents_normal.xlsx"
    df = pd.read_excel(file1)
    df2 = pd.read_excel(file2)
    df2['timestamp'] = pd.to_datetime(df['timestamp'])
    df2.set_index('timestamp', inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    ax.plot(df2['Plugged In Agents'], label='Scenario 13 - Plugged In Agents', color='olive', linestyle='solid')
    ax.plot(df2['Charging Agents'], label='Scenario 13 - Charging Agents', color='olive', linestyle=(0, (1, 1)))

    ax.plot(df['Plugged In Agents'], label='Scenario 15 - Plugged In Agents', color='black', linestyle='solid')
    ax.plot(df['Charging Agents'], label='Scenario 15 - Charging Agents', color='black', linestyle=(0, (1, 1)))

    plt.title('Charging and Plugged in Agents - Scenario 13 and 15')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.xlim(df.index.min(), df.index.max())
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), ncol=1, frameon=False)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()

    plt.show()



if __name__ == '__main__':
    # plot_agent_min_mean_max_recharge_power()
    # calculate_agent_statistics()
    # read_agent_statistics_results()  # TODO implement good statistics to show that car has enough time to charge
    # charging_priority_one_agent_example()   # create one example plot for charging priority
    # plot_one_agent_charging_load()
    # spitzenlast_pro_auto()
    # dump()   # To create a file dump of all agent data of one simulation run for one scenario
    # create_agent_plot()
    # read_agent_statistics_recharge_sum()
    # dump()
    plugged_in_charging_agents()