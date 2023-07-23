from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FixedLocator, MultipleLocator, FixedFormatter, MaxNLocator, LogLocator
import matplotlib.dates as mdates


def create_plot(start_date, end_date, df_results: pd.DataFrame, title):
    # # black and white
    # plt.style.use('grayscale')
    x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
    x_axis_time = x_axis_time[:-1]

    df_results['timestamp'] = x_axis_time
    df_results.set_index('timestamp', inplace=True)

    fig, ax = plt.subplots()

    df_results.plot(y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'], ax=ax)

    plt.xlabel('Timestamp')
    plt.ylabel('kW')

    ax.set_xticks(df_results.index[::24])
    ax.set_xticklabels(df_results.index[::24].strftime('%d-%m %H:%M'), rotation=90)

    lines = ax.get_lines()

    linestyles = ['-.', '--', ':', '-']
    for i, line in enumerate(lines):
        line.set_linestyle(linestyles[i % len(linestyles)])

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)

    legend.get_frame().set_facecolor('white')

    plt.title(title)
    plt.subplots_adjust(bottom=0.3)

    plt.tight_layout()
    plt.show()

def plot_all():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    csv_files = list(directory.glob("*025_interaction_*/results_all_runs.csv"))
    csv_files = sorted(csv_files)

    num_subplots = len(csv_files)

    num_rows = (num_subplots + 1) // 2
    num_cols = min(2, num_subplots)

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)

    unique_labels = set()

    row_y_limits = []

    lines = []  # Store lines for creating a common legend

    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
        x_axis_time = x_axis_time[:-1]

        df['timestamp'] = x_axis_time
        df.set_index('timestamp', inplace=True)

        # Select the subplot for the current file
        ax = fig.add_subplot(gs[i])

        # Plot the desired columns from the DataFrame on the subplot
        lines += df.plot(y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'],
                         ax=ax).lines

        # Customize subplot properties if needed
        ax.set_title(csv_file.parent.name)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('kW')

        # Get the legend labels for the current subplot
        labels = [line.get_label() for line in ax.get_lines()]

        # Add unique labels to the set of unique legend labels
        unique_labels.update(labels)

        # Set the same y-axis limits for plots in the same row
        if i % num_cols == 0 and i + 1 < num_subplots:
            row_y_limits.append(ax.get_ylim())

        elif i % num_cols == 1:
            ax.set_ylim(row_y_limits[-1])

        ax.set_xticks(df.index[::96])
        ax.set_xticklabels(df.index[::96].strftime('%d-%m'), rotation=90)

        # Remove x-axis tick labels except for the last row
        if i // num_cols != num_rows - 1:
            ax.set_xticks([])
            ax.set_xticklabels([])

        ax.get_legend().remove()

    # Set legend properties
    # legend_labels = list(unique_labels)
    # fig.legend(lines, legend_labels, loc='lower center', ncol=2, frameon=False)

    # Adjust spacing between subplots
    gs.tight_layout(fig)

    # Display the plot
    plt.show()


# create plot with all results of model runs
def create_diff_plots_all_runs_interaction():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    scenarios = [
        # "*_025_interaction_false_norm*/results_all_runs.csv",
        "*_025_interaction_true_norm*/results_all_runs.csv",
        # "*_025_interaction_false_off*/results_all_runs.csv",
        "*_025_interaction_true_off*/results_all_runs.csv",

        # "*_050_interaction_false_norm*/results_all_runs.csv",
        "*_050_interaction_true_norm*/results_all_runs.csv",
        # "*_050_interaction_false_off*/results_all_runs.csv",
        "*_050_interaction_true_off*/results_all_runs.csv",

        # "*_150_interaction_false_norm*/results_all_runs.csv",
        "*_150_interaction_true_norm*/results_all_runs.csv",
        # "*_150_interaction_false_off*/results_all_runs.csv",
        "*_150_interaction_true_off*/results_all_runs.csv",

        # "*_300_interaction_false_norm*/results_all_runs.csv",
        "*_300_interaction_true_norm*/results_all_runs.csv",
        # "*_300_interaction_false_off*/results_all_runs.csv",
        "*_300_interaction_true_off*/results_all_runs.csv"
    ]
    titles = ['Scenario 2', 'Scenario 4', 'Scenario 6', 'Scenario 8', 'Scenario 10', 'Scenario 12', 'Scenario 14', 'Scenario 16']

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    # List to store all CSV file paths
    csv_files = []

    # Iterate over each pattern and find matching CSV files
    for pattern in scenarios:
        matching_files = directory.glob(pattern)
        csv_files.extend(matching_files)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12))
    lines_last_subplot = []
    row_y_limits = []

    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        name = str(csv_file)
        if '025' in name:
            df = df / 25
        elif '050' in name:
            df = df / 50
        elif '150' in name:
            df = df / 150
        elif '300' in name:
            df = df / 300

        x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
        x_axis_time = x_axis_time[:-1]

        df['timestamp'] = x_axis_time
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the 'timestamp' column as the index
        df.set_index('timestamp', inplace=True)

        # Extract the hour and minute components from the timestamp
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        # Group the data by hour and quarter hour and calculate the average
        df_grouped = df.groupby(['hour', 'minute']).mean()

        # Reset the index to turn the grouped columns into regular columns
        df_grouped.reset_index(inplace=True)

        # Remove the unnecessary columns
        df_grouped = df_grouped[
            ['hour', 'minute', 'total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity']]
        row = i // 2  # Calculate the row index
        col = i % 2  # Calculate the column index
        ax = axes[row, col]  # Select the subplot

        df_grouped['minute'] = df_grouped['minute'].astype(str).str.zfill(2)
        df_grouped['time'] = df_grouped['hour'].astype(str) + ':' + df_grouped['minute'].astype(str)

        custom_palette = ["blue", "blue", "blue", "#008000"]
        linestyles = ['dotted', 'dashed', 'solid', 'solid']

        ax.plot(df_grouped['time'], df_grouped['transformer_capacity'], linestyle=linestyles[3],
                label='Transformer Capacity 100%', color=custom_palette[3])

        ax.plot(df_grouped['time'], df_grouped['transformer_capacity'] * 1.5, linestyle=linestyles[3],
                label='Transformer Capacity 150 %', color='#ff0000')

        # ax.plot(df_grouped['time'], df_grouped['transformer_capacity'] * 2, linestyle=linestyles[3],
        #         label='Transformer Capacity 200 %', color='#800080')

        ax.plot(df_grouped['time'], df_grouped['total_recharge_power'], linestyle=linestyles[0], label='Charging Load', color=custom_palette[0])
        ax.plot(df_grouped['time'], df_grouped['total_customer_load'], linestyle=linestyles[1], label='Customer Load', color=custom_palette[1])
        ax.plot(df_grouped['time'], df_grouped['total_load'], linestyle=linestyles[2], label='Total Load', color=custom_palette[2])



        # df_grouped.plot(x='time',
        #                 y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'],
        #                 color=custom_palette,
        #                 linestyle=linestyles,
        #                 ax=ax)

        # df_grouped.plot(y='total_recharge_power', ax=ax)
        tick_positions = df_grouped.index[::12]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(df_grouped['time'][::12], rotation=90)

        ax.set_title(titles[i])
        ax.set_ylabel('Charging Power \n [kW]')
        ax.legend().remove()
        ax.set_xlim(0, 96)
        if i == len(csv_files) - 1:
            # Store lines from the last subplot
            lines_last_subplot = ax.get_lines()

    axes[-1, -1].set_xlabel('Hours')
    axes[-1, -2].set_xlabel('Hours')
    # axes[-1, -3].set_xlabel('Hours')
    # axes[-1, -4].set_xlabel('Hours')

    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    # axes[0, 2].set_xticklabels([])
    # axes[0, 3].set_xticklabels([])

    axes[1, 0].set_xticklabels([])
    axes[1, 1].set_xticklabels([])
    # axes[1, 2].set_xticklabels([])
    # axes[1, 3].set_xticklabels([])

    axes[2, 0].set_xticklabels([])
    axes[2, 1].set_xticklabels([])
    # axes[2, 2].set_xticklabels([])
    # axes[2, 3].set_xticklabels([])

    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    # axes[0, 2].set_xlabel('')
    # axes[0, 3].set_xlabel('')

    axes[1, 0].set_xlabel('')
    axes[1, 1].set_xlabel('')
    # axes[1, 2].set_xlabel('')
    # axes[1, 3].set_xlabel('')

    axes[2, 0].set_xlabel('')
    axes[2, 1].set_xlabel('')
    # axes[2, 2].set_xlabel('')
    # axes[2, 3].set_xlabel('')

    # Create a single legend below the plot using lines from the last subplot
    legend = plt.figlegend(lines_last_subplot, [line.get_label() for line in lines_last_subplot],
                           loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.00), frameon=False)
    # legend.get_frame().set_facecolor('white')
    plt.subplots_adjust(bottom=0.17, wspace=0.35, hspace=0.5)
    plt.savefig('all_runs_one_plot_interaction.png', dpi=300)
    # plt.tight_layout()
    plt.show()


# create plot with all results of model runs
def create_diff_plots_all_runs_no_interaction():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    scenarios = [
        "*_025_interaction_false_norm*/results_all_runs.csv",
        # "*_025_interaction_true_norm*/results_all_runs.csv",
        "*_025_interaction_false_off*/results_all_runs.csv",
        # "*_025_interaction_true_off*/results_all_runs.csv",

        "*_050_interaction_false_norm*/results_all_runs.csv",
        # "*_050_interaction_true_norm*/results_all_runs.csv",
        "*_050_interaction_false_off*/results_all_runs.csv",
        # "*_050_interaction_true_off*/results_all_runs.csv",

        "*_150_interaction_false_norm*/results_all_runs.csv",
        # "*_150_interaction_true_norm*/results_all_runs.csv",
        "*_150_interaction_false_off*/results_all_runs.csv",
        # "*_150_interaction_true_off*/results_all_runs.csv",

        "*_300_interaction_false_norm*/results_all_runs.csv",
        # "*_300_interaction_true_norm*/results_all_runs.csv",
        "*_300_interaction_false_off*/results_all_runs.csv",
        # "*_300_interaction_true_off*/results_all_runs.csv"
    ]
    titles = ['Scenario 1', 'Scenario 3', 'Scenario 5', 'Scenario 7', 'Scenario 9', 'Scenario 11', 'Scenario 13', 'Scenario 15']

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    # List to store all CSV file paths
    csv_files = []

    # Iterate over each pattern and find matching CSV files
    for pattern in scenarios:
        matching_files = directory.glob(pattern)
        csv_files.extend(matching_files)

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12))
    lines_last_subplot = []
    row_y_limits = []

    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        name = str(csv_file)
        if '025' in name:
            df = df / 25
        elif '050' in name:
            df = df / 50
        elif '150' in name:
            df = df / 150
        elif '300' in name:
            df = df / 300

        x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
        x_axis_time = x_axis_time[:-1]

        df['timestamp'] = x_axis_time
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the 'timestamp' column as the index
        df.set_index('timestamp', inplace=True)

        # Extract the hour and minute components from the timestamp
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        # Group the data by hour and quarter hour and calculate the average
        df_grouped = df.groupby(['hour', 'minute']).mean()

        # Reset the index to turn the grouped columns into regular columns
        df_grouped.reset_index(inplace=True)

        # Remove the unnecessary columns
        df_grouped = df_grouped[
            ['hour', 'minute', 'total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity']]
        row = i // 2  # Calculate the row index
        col = i % 2  # Calculate the column index
        ax = axes[row, col]  # Select the subplot

        df_grouped['minute'] = df_grouped['minute'].astype(str).str.zfill(2)
        df_grouped['time'] = df_grouped['hour'].astype(str) + ':' + df_grouped['minute'].astype(str)

        custom_palette = ["blue", "blue", "blue", "#008000"]
        linestyles = ['dotted', 'dashed', 'solid', 'solid']

        ax.plot(df_grouped['time'], df_grouped['transformer_capacity'], linestyle=linestyles[3],
                label='Transformer Capacity 100%', color=custom_palette[3])

        ax.plot(df_grouped['time'], df_grouped['transformer_capacity'] * 1.5, linestyle=linestyles[3],
                label='Transformer Capacity 150 %', color='#ff0000')

        # ax.plot(df_grouped['time'], df_grouped['transformer_capacity'] * 2, linestyle=linestyles[3],
        #         label='Transformer Capacity 200 %', color='#800080')

        ax.plot(df_grouped['time'], df_grouped['total_recharge_power'], linestyle=linestyles[0], label='Charging Load', color=custom_palette[0])
        ax.plot(df_grouped['time'], df_grouped['total_customer_load'], linestyle=linestyles[1], label='Customer Load', color=custom_palette[1])
        ax.plot(df_grouped['time'], df_grouped['total_load'], linestyle=linestyles[2], label='Total Load', color=custom_palette[2])



        # df_grouped.plot(x='time',
        #                 y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'],
        #                 color=custom_palette,
        #                 linestyle=linestyles,
        #                 ax=ax)

        # df_grouped.plot(y='total_recharge_power', ax=ax)
        tick_positions = df_grouped.index[::12]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(df_grouped['time'][::12], rotation=90)

        ax.set_title(titles[i])
        ax.set_ylabel('Charging Power \n [kW]')
        ax.legend().remove()
        ax.set_xlim(0, 96)
        if i == len(csv_files) - 1:
            # Store lines from the last subplot
            lines_last_subplot = ax.get_lines()

    axes[-1, -1].set_xlabel('Hours')
    axes[-1, -2].set_xlabel('Hours')
    # axes[-1, -3].set_xlabel('Hours')
    # axes[-1, -4].set_xlabel('Hours')

    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    # axes[0, 2].set_xticklabels([])
    # axes[0, 3].set_xticklabels([])

    axes[1, 0].set_xticklabels([])
    axes[1, 1].set_xticklabels([])
    # axes[1, 2].set_xticklabels([])
    # axes[1, 3].set_xticklabels([])

    axes[2, 0].set_xticklabels([])
    axes[2, 1].set_xticklabels([])
    # axes[2, 2].set_xticklabels([])
    # axes[2, 3].set_xticklabels([])

    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    # axes[0, 2].set_xlabel('')
    # axes[0, 3].set_xlabel('')

    axes[1, 0].set_xlabel('')
    axes[1, 1].set_xlabel('')
    # axes[1, 2].set_xlabel('')
    # axes[1, 3].set_xlabel('')

    axes[2, 0].set_xlabel('')
    axes[2, 1].set_xlabel('')
    # axes[2, 2].set_xlabel('')
    # axes[2, 3].set_xlabel('')

    # Create a single legend below the plot using lines from the last subplot
    legend = plt.figlegend(lines_last_subplot, [line.get_label() for line in lines_last_subplot],
                           loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.00), frameon=False)
    # legend.get_frame().set_facecolor('white')
    plt.subplots_adjust(bottom=0.17, wspace=0.35, hspace=0.5)
    plt.savefig('all_runs_one_plot_no_interaction.png', dpi=300)
    # plt.tight_layout()
    plt.show()


def average_load_profiles_models_one_agent():
    start_date = '2008-07-13'
    end_date = '2008-07-27'
    title = 'Average Charging Profile\nScenario 4, 8, 12, 16'
    scenarios = [
        # "*_025_interaction_false_norm*/results_run_*_model*.csv",
        # "*_025_interaction_true_norm*/results_run_*_model*.csv",
        # "*_025_interaction_false_off*/results_run_*_model*.csv",
        "*_025_interaction_true_off*/results_run_*_model*.csv",

        # "*_050_interaction_false_norm*/results_run_*_model*.csv",
        # "*_050_interaction_true_norm*/results_run_*_model*.csv",
        # "*_050_interaction_false_off*/results_run_*_model*.csv",
        "*_050_interaction_true_off*/results_run_*_model*.csv",

        # "*_150_interaction_false_norm*/results_run_*_model*.csv",
        # "*_150_interaction_true_norm*/results_run_*_model*.csv",
        # "*_150_interaction_false_off*/results_run_*_model*.csv",
        "*_150_interaction_true_off*/results_run_*_model*.csv",

        # "*_300_interaction_false_norm*/results_run_*_model*.csv",
        # "*_300_interaction_true_norm*/results_run_*_model*.csv",
        # "*_300_interaction_false_off*/results_run_*_model*.csv",
        "*_300_interaction_true_off*/results_run_*_model*.csv"
    ]

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    list_of_df_customer_load = []
    list_of_df_recharge_power = []

    for j, scenario in enumerate(scenarios):

        csv_files = list(directory.glob(scenario))
        csv_files = sorted(csv_files)

        for i, csv_file in enumerate(csv_files):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            name = str(csv_file)
            if '025' in name:
                df = df / 25
            elif '050' in name:
                df = df / 50
            elif '150' in name:
                df = df / 150
            elif '300' in name:
                df = df / 300

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df['timestamp'] = x_axis_time
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Set the 'timestamp' column as the index
            df.set_index('timestamp', inplace=True)
            list_of_df_recharge_power.append(df['total_recharge_power'])
            list_of_df_customer_load.append(df['total_customer_load'])

    result_df_recharge_power = pd.concat(list_of_df_recharge_power, axis=1)

    result_df_recharge_power['average'] = result_df_recharge_power['total_recharge_power'].mean(axis=1)
    result_df_recharge_power['percentile_5'] = result_df_recharge_power['total_recharge_power'].quantile(0.05, axis=1)
    result_df_recharge_power['percentile_95'] = result_df_recharge_power['total_recharge_power'].quantile(0.95, axis=1)
    # result_df_recharge_power.set_index('timestamp', inplace=True)

    # Extract the hour and minute components from the timestamp
    result_df_recharge_power['hour'] = result_df_recharge_power.index.hour
    result_df_recharge_power['minute'] = result_df_recharge_power.index.minute

    # Group the data by hour and quarter hour and calculate the average
    df_grouped = result_df_recharge_power.groupby(['hour', 'minute']).mean()

    # Reset the index to turn the grouped columns into regular columns
    df_grouped.reset_index(inplace=True)

    # Remove the unnecessary columns
    df_grouped = df_grouped[['hour', 'minute', 'average', 'percentile_5', 'percentile_95']]

    df_grouped['minute'] = df_grouped['minute'].astype(str).str.zfill(2)
    df_grouped['time'] = df_grouped['hour'].astype(str) + ':' + df_grouped['minute'].astype(str)

    df_grouped.plot(x='time', y='average', ax=ax, label='Average Charging Profile', color='blue', linestyle='dotted', linewidth=1.5)
    df_grouped.plot(x='time', y='percentile_5', ax=ax, label='5% Percentile', color='darkgrey', linewidth=0.5)
    df_grouped.plot(x='time', y='percentile_95', ax=ax, label='95% Percentile', color='darkgrey', linewidth=0.5)

    ax.fill_between(df_grouped['time'], df_grouped['percentile_5'], df_grouped['percentile_95'], color='grey',
                    alpha=0.1, label='5% - 95% Percentile')

    ax.set_title(title)
    ax.set_ylabel('Charging Power\n[kW]')
    ax.legend().remove()
    ax.set_xlim(0, 96)
    # ax.set_ylim(0, 1.2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    tick_positions = df_grouped.index[::4]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(df_grouped['time'][::4], rotation=90)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    fig_name = title.replace('\n', "")
    plt.savefig(fig_name, dpi=300)
    plt.show()

def combination_of_average_load_profiles():
    start_date = '2008-07-13'
    end_date = '2008-07-27'
    title = 'Average Load Profile\nCombination of Scenario 14, 16'
    scenarios = [
        # "*_025_interaction_false_norm*/results_run_*_model*.csv",
        # "*_025_interaction_true_norm*/results_run_*_model*.csv",
        # "*_025_interaction_false_off*/results_run_*_model*.csv",
        # "*_025_interaction_true_off*/results_run_*_model*.csv",

        # "*_050_interaction_false_norm*/results_run_*_model*.csv",
        # "*_050_interaction_true_norm*/results_run_*_model*.csv",
        # "*_050_interaction_false_off*/results_run_*_model*.csv",
        # "*_050_interaction_true_off*/results_run_*_model*.csv",

        # "*_150_interaction_false_norm*/results_run_*_model*.csv",
        # "*_150_interaction_true_norm*/results_run_*_model*.csv",
        # "*_150_interaction_false_off*/results_run_*_model*.csv",
        # "*_150_interaction_true_off*/results_run_*_model*.csv",

        # "*_300_interaction_false_norm*/results_run_*_model*.csv",
        "*_300_interaction_true_norm*/results_run_*_model*.csv",
        # "*_300_interaction_false_off*/results_run_*_model*.csv",
        "*_300_interaction_true_off*/results_run_*_model*.csv"
    ]

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    list_of_df_customer_load = []
    list_of_df_recharge_power = []
    list_of_df_total_load = []

    for j, scenario in enumerate(scenarios):

        csv_files = list(directory.glob(scenario))
        csv_files = sorted(csv_files)

        for i, csv_file in enumerate(csv_files):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            name = str(csv_file)
            if '025' in name:
                df = df / 25
            elif '050' in name:
                df = df / 50
            elif '150' in name:
                df = df / 150
            elif '300' in name:
                df = df / 300

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df['timestamp'] = x_axis_time
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Set the 'timestamp' column as the index
            df.set_index('timestamp', inplace=True)
            list_of_df_recharge_power.append(df['total_recharge_power'])
            list_of_df_customer_load.append(df['total_customer_load'])
            list_of_df_total_load.append(df['total_load'])

    result_df_recharge_power = pd.concat(list_of_df_recharge_power, axis=1)
    result_df_customer_load = pd.concat(list_of_df_customer_load, axis=1)

    copy_1 = result_df_recharge_power.copy()
    copy_2 = result_df_customer_load.copy()

    copy_1.columns = copy_2.columns
    copy_1.index = copy_2.index

    both_df = copy_1.add(copy_2, axis=1)
    both_df = both_df.rename(columns=lambda col: 'total_load')
    both_df['average_total'] = both_df['total_load'].mean(axis=1)

    result_df_customer_load['average_customer'] = result_df_customer_load['total_customer_load'].mean(axis=1)
    result_df_recharge_power['average_car'] = result_df_recharge_power['total_recharge_power'].mean(axis=1)

    both_df['percentile_5'] = both_df['total_load'].quantile(0.05, axis=1)
    both_df['percentile_95'] = both_df['total_load'].quantile(0.95, axis=1)
    both_df['hour'] = result_df_recharge_power.index.hour
    both_df['minute'] = result_df_recharge_power.index.minute

    result_df_recharge_power['percentile_5'] = result_df_recharge_power['total_recharge_power'].quantile(0.05, axis=1)
    result_df_recharge_power['percentile_95'] = result_df_recharge_power['total_recharge_power'].quantile(0.95, axis=1)
    # result_df_recharge_power.set_index('timestamp', inplace=True)

    # Extract the hour and minute components from the timestamp
    result_df_recharge_power['hour'] = result_df_recharge_power.index.hour
    result_df_recharge_power['minute'] = result_df_recharge_power.index.minute

    result_df_customer_load['hour'] = result_df_recharge_power.index.hour
    result_df_customer_load['minute'] = result_df_recharge_power.index.minute

    # Group the data by hour and quarter hour and calculate the average
    df_grouped = result_df_recharge_power.groupby(['hour', 'minute']).mean()
    df_grouped_customer = result_df_customer_load.groupby(['hour', 'minute']).mean()
    both_df = both_df.groupby(['hour', 'minute']).mean()
    # Reset the index to turn the grouped columns into regular columns
    df_grouped.reset_index(inplace=True)
    df_grouped_customer.reset_index(inplace=True)
    both_df.reset_index(inplace=True)

    # Remove the unnecessary columns
    df_grouped = df_grouped[['hour', 'minute', 'average_car', 'percentile_5', 'percentile_95']]
    df_grouped_customer = df_grouped_customer[['hour', 'minute', 'average_customer']]
    both_df = both_df[['hour', 'minute', 'average_total', 'percentile_5', 'percentile_95']]

    both_df['minute'] = both_df['minute'].astype(str).str.zfill(2)
    both_df['time'] = both_df['hour'].astype(str) + ':' + both_df['minute'].astype(str)

    df_grouped['minute'] = df_grouped['minute'].astype(str).str.zfill(2)
    df_grouped['time'] = df_grouped['hour'].astype(str) + ':' + df_grouped['minute'].astype(str)

    df_grouped_customer['minute'] = df_grouped_customer['minute'].astype(str).str.zfill(2)
    df_grouped_customer['time'] = df_grouped_customer['hour'].astype(str) + ':' + df_grouped_customer['minute'].astype(str)

    df_grouped.plot(x='time', y='average_car', ax=ax, label='Average Charging Load', color='blue', linestyle='dotted', linewidth=1.5)
    df_grouped_customer.plot(x='time', y='average_customer', ax=ax, label='Average Customer Load', color='blue', linestyle='dashed',
                             linewidth=1.5)
    both_df.plot(x='time', y='average_total', ax=ax, label='Average Total Load', color='blue', linewidth=1.5)

    both_df.plot(x='time', y='percentile_5', ax=ax, label='5% Percentile', color='darkgrey', linewidth=0.5)
    both_df.plot(x='time', y='percentile_95', ax=ax, label='95% Percentile', color='darkgrey', linewidth=0.5)
    ax.fill_between(both_df['time'], both_df['percentile_5'], both_df['percentile_95'], color='grey',
                    alpha=0.1, label='5% - 95% Percentile')

    ax.set_title(title)
    ax.set_ylabel('Charging Power\n[kW]')
    ax.legend().remove()
    ax.set_xlim(0, 96)
    # ax.set_ylim(0, 1.2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    tick_positions = df_grouped.index[::4]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(df_grouped['time'][::4], rotation=90)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    fig_name = title.replace('\n', "")
    plt.savefig(fig_name, dpi=300)
    plt.show()


def all_model_results_and_average():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    scenarios = [
        "*_025_interaction_false_norm*/results_run_*_model*.csv",
        "*_025_interaction_true_norm*/results_run_*_model*.csv",
        "*_025_interaction_false_off*/results_run_*_model*.csv",
        "*_025_interaction_true_off*/results_run_*_model*.csv",

        "*_050_interaction_false_norm*/results_run_*_model*.csv",
        "*_050_interaction_true_norm*/results_run_*_model*.csv",
        "*_050_interaction_false_off*/results_run_*_model*.csv",
        "*_050_interaction_true_off*/results_run_*_model*.csv",

        "*_150_interaction_false_norm*/results_run_*_model*.csv",
        "*_150_interaction_true_norm*/results_run_*_model*.csv",
        "*_150_interaction_false_off*/results_run_*_model*.csv",
        "*_150_interaction_true_off*/results_run_*_model*.csv",

        "*_300_interaction_false_norm*/results_run_*_model*.csv",
        "*_300_interaction_true_norm*/results_run_*_model*.csv",
        "*_300_interaction_false_off*/results_run_*_model*.csv",
        "*_300_interaction_true_off*/results_run_*_model*.csv"
    ]

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    for j, scenario in enumerate(scenarios):
        csv_files = list(directory.glob(scenario))
        csv_files = sorted(csv_files)

        list_of_df_recharge_power = []
        list_of_df_customer_load = []
        list_of_df_transformer_capacity = []
        list_of_df_load = []

        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(csv_file)
            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df['timestamp'] = x_axis_time
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Set the 'timestamp' column as the index
            df.set_index('timestamp', inplace=True)

            list_of_df_recharge_power.append(df['total_recharge_power'])
            list_of_df_customer_load.append(df['total_customer_load'])
            list_of_df_transformer_capacity.append(df['transformer_capacity'])
            list_of_df_load.append(df['total_load'])

        result_df_recharge_power = pd.concat(list_of_df_recharge_power, axis=1)
        result_df_recharge_power['average'] = result_df_recharge_power['total_recharge_power'].mean(axis=1)
        result_df_recharge_power['percentile_5'] = result_df_recharge_power['total_recharge_power'].quantile(0.05, axis=1)
        result_df_recharge_power['percentile_95'] = result_df_recharge_power['total_recharge_power'].quantile(0.95, axis=1)

        # Concatenate the dataframes from 'list_of_df_total_load'
        result_df_total_load = pd.concat(list_of_df_load, axis=1)

        # Calculate the average, 5th percentile, and 95th percentile for each row
        result_df_total_load['average'] = result_df_total_load.mean(axis=1)
        result_df_total_load['percentile_5'] = result_df_total_load.quantile(0.05, axis=1)
        result_df_total_load['percentile_95'] = result_df_total_load.quantile(0.95, axis=1)

        name = str(csv_file)

        x = result_df_total_load.index
        y = result_df_total_load['average']
        y1 = result_df_total_load['percentile_5']
        y2 = result_df_total_load['percentile_95']
        y3 = list_of_df_transformer_capacity[0]
        y4 = list_of_df_transformer_capacity[0] * 1.5

        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

        # Plot the data on both subplots
        ax.plot(x, y, label='Average Total Load', color='blue', linewidth=1.5)
        ax.plot(x, y1, label='5% Percentile', color='darkgrey', linewidth=0.5)
        ax.plot(x, y2, label='95% Percentile', color='darkgrey', linewidth=0.5)
        ax.plot(x, y3, label='Transformer Capacity 100%', color='#008000')
        ax.plot(x, y4, label='Transformer Capacity 150%', color='#ff0000')
        ax.fill_between(result_df_total_load.index, y1, y2, color='grey', alpha=0.1, label='5% - 95% Percentile')

        ax2.plot(x, y, label='Average Total Load', color='blue', linewidth=2)
        ax2.plot(x, y1, label='5% Percentile', color='darkgrey', linewidth=0.5)
        ax2.plot(x, y2, label='95% Percentile', color='darkgrey', linewidth=0.5)
        ax2.plot(x, y3, label='Transformer Capacity 100%', color='#008000')
        ax2.plot(x, y4, label='Transformer Capacity 150%', color='#ff0000')
        ax2.fill_between(result_df_total_load.index, y1, y2, color='grey', alpha=0.1, label='5%-95% Percentile')

        # Customize the y-axis limits for different scenarios
        if '025' in name:
            ax.set_ylim(40, 200)
            ax2.set_ylim(0, 40)
        elif '050' in name:
            ax.set_ylim(75, 350)
            ax2.set_ylim(0, 75)
        elif '150' in name:
            ax.set_ylim(250, 1000)  # outliers only
            ax2.set_ylim(0, 250)  # most of the data
        elif '300' in name:
            ax.set_ylim(350, 1800)
            ax2.set_ylim(0, 350)

        # Customize plot appearance
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(top=False, labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        plt.xticks(rotation=90)

        d = 0.015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax.transAxes, linewidth=0.5, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

        title = "Scenario " + str(j + 1)
        f.suptitle(title)
        f.text(0.02, 0.55, 'Charging Power [kW]', va='center', rotation='vertical')
        # f.text(0.5, 0.2, 'Date', ha='center')

        spacing = 0.005
        ax.set_position([0.125, 0.6, 0.775, 0.25])
        ax2.set_position([0.125, 0.6 - ax.get_position().height - spacing, 0.775, 0.25])

        handles, labels = ax.get_legend_handles_labels()
        f.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.09))

        plt.xlim(result_df_total_load.index.min(), result_df_total_load.index.max())  # Set the x-axis limits
        fig_name = title + '_weekly_profile_total'
        plt.savefig(fig_name, dpi=300)
        plt.show()


def print_model_run_one_plot():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    csv_files = list(directory.glob("*_interaction_*/results_all_runs.csv"))
    csv_files = sorted(csv_files)

    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        name = str(csv_file)

        x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
        x_axis_time = x_axis_time[:-1]

        df['timestamp'] = x_axis_time
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the 'timestamp' column as the index
        df.set_index('timestamp', inplace=True)

        x = df.index
        y = df['total_recharge_power']
        y1 = df['total_customer_load']
        y2 = df['total_load']
        y3 = df['transformer_capacity']

        f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

        ax.plot(x, y, label='total_recharge_power')
        ax.plot(x, y1, label='total_customer_load')
        ax.plot(x, y2, label='total_load')
        ax.plot(x, y3, label='transformer_capacity')

        ax2.plot(x, y, label='total_recharge_power')
        ax2.plot(x, y1, label='total_customer_load')
        ax2.plot(x, y2, label='total_load')
        ax2.plot(x, y3, label='transformer_capacity')

        # zoom-in / limit the view to different portions of the data       if '025' in name:
        if '025' in name:
            ax.set_ylim(40, 150)
            ax2.set_ylim(0, 40)

        elif '050' in name:
            ax.set_ylim(75, 300)
            ax2.set_ylim(0, 75)

        elif '150' in name:
            ax.set_ylim(250, 1700)  # outliers only
            ax2.set_ylim(0, 250)  # most of the data

        elif '300' in name:
            ax.set_ylim(350, 1700)
            ax2.set_ylim(0, 350)

        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(top=False, labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()
        plt.xticks(rotation=90)

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, linewidth=0.5, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

        title = "Scenario " + str(i + 1)
        f.suptitle(title)
        f.text(0.02, 0.55, 'Charging Power [kW]', va='center', rotation='vertical')
        f.text(0.5, 0.2, 'Date', ha='center')

        spacing = 0.005
        ax.set_position([0.125, 0.6, 0.775, 0.25])
        ax2.set_position([0.125, 0.6 - ax.get_position().height - spacing, 0.775, 0.25])

        handles, labels = ax.get_legend_handles_labels()
        f.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.09))


        # plt.ylim(0, 6)
        # TODO Turn this on
        plt.xlim(df.index.min(), df.index.max())
        # ax.set_xlim(0, 14)

        # fig_name = title + '_weekly_profile_total'
        # plt.savefig(fig_name, dpi=300)
        plt.show()


def create_heat_map():
    # create a heat map model
    # yaxis time
    # xaxis day
    # heat = percentage of transformer capacity needed by recharging power
    # Create a figure and axes
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    scenarios = [
        "*_025_interaction_false_norm*/results_run_*_model*.csv",
        "*_025_interaction_true_norm*/results_run_*_model*.csv",
        "*_025_interaction_false_off*/results_run_*_model*.csv",
        "*_025_interaction_true_off*/results_run_*_model*.csv",

        "*_050_interaction_false_norm*/results_run_*_model*.csv",
        "*_050_interaction_true_norm*/results_run_*_model*.csv",
        "*_050_interaction_false_off*/results_run_*_model*.csv",
        "*_050_interaction_true_off*/results_run_*_model*.csv",

        "*_150_interaction_false_norm*/results_run_*_model*.csv",
        "*_150_interaction_true_norm*/results_run_*_model*.csv",
        "*_150_interaction_false_off*/results_run_*_model*.csv",
        "*_150_interaction_true_off*/results_run_*_model*.csv",

        "*_300_interaction_false_norm*/results_run_*_model*.csv",
        "*_300_interaction_true_norm*/results_run_*_model*.csv",
        "*_300_interaction_false_off*/results_run_*_model*.csv",
        "*_300_interaction_true_off*/results_run_*_model*.csv"
    ]

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    factor = 1.0
    only_above = True

    for j, scenario in enumerate(scenarios):
        csv_files = list(directory.glob(scenario))
        print(len(csv_files))

        df_results_total = []
        # Append all to one
        for i, csv_file in enumerate(csv_files):
            df_results = pd.read_csv(csv_file)

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df_results['timestamp'] = x_axis_time
            df_results.set_index('timestamp', inplace=True)
            # df_results['total_recharge_power'] = df_results['total_recharge_power']
            # df_results['total_load'] = df_results['total_recharge_power'] + df_results['total_customer_load']
            df_results_total.append(df_results)

        concatenated_df = pd.concat(df_results_total, ignore_index=False)
        if only_above:
            concatenated_df['overloaded'] = concatenated_df.apply(
                lambda row: 1 if round(row['total_load'], 2) > (row['transformer_capacity'] * factor) else 0, axis=1)
        else:
            concatenated_df['overloaded'] = concatenated_df.apply(
                lambda row: 1 if round(row['total_load'], 2) >= (row['transformer_capacity'] * factor) else 0, axis=1)

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        df_agg = (concatenated_df.groupby([concatenated_df.index.strftime('%A'), concatenated_df.index.strftime('%H:%M')])
                    ['overloaded']
                    .sum()
                    .unstack(level=0)
            )

        df_agg = df_agg[day_order]
        # divide by 60 because this is the maximum that can happen
        df_agg = df_agg.div(60).multiply(100)
        # df_agg['overloaded'] = df_agg['overloaded'] / 25
        import seaborn as sns
        # Create the heatmap using seaborn

        heatmap = sns.heatmap(df_agg, cmap='hot', vmin=0, vmax=100)

        # Set x-axis and y-axis labels
        plt.ylabel('Time')
        plt.xlabel('')

        if factor == 1:
            title = 'Scenario {}\nTransformer Full Capacity or Above'.format(j + 1)

        if factor == 1 and only_above:
            title = 'Scenario {}\nTransformer Capacity Over 100 %'.format(j + 1)

        if factor == 1.25:
            title = 'Scenario {}\nTransformer Capacity Over 125 %'.format(j + 1)

        if factor == 1.5:
            title = 'Scenario {}\nTransformer Capacity Over 150 %'.format(j + 1)

        if factor == 2:
            title = 'Scenario {}\nTransformer Capacity Over 200 %'.format(j + 1)

        # Set the title
        plt.title(title)

        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)

        cax = heatmap.collections[0].colorbar.ax
        cax.spines['top'].set_visible(True)
        cax.spines['bottom'].set_visible(True)
        cax.spines['left'].set_visible(True)
        cax.spines['right'].set_visible(True)

        # Display the plot
        plt.tight_layout()
        plt.savefig('Scenario {}_Heatmap_{}.png'.format(j + 1, factor), dpi=300)
        plt.clf()



def prozentuale_ueberlastung(split=False):
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    if split:
        scenarios = [
            "*_025_interaction_false_norm*/results_run_*_model*.csv",
            # "*_025_interaction_true_norm*/results_run_*_model*.csv",
            "*_025_interaction_false_off*/results_run_*_model*.csv",
            # "*_025_interaction_true_off*/results_run_*_model*.csv",

            "*_050_interaction_false_norm*/results_run_*_model*.csv",
            # "*_050_interaction_true_norm*/results_run_*_model*.csv",
            "*_050_interaction_false_off*/results_run_*_model*.csv",
            # "*_050_interaction_true_off*/results_run_*_model*.csv",

            "*_150_interaction_false_norm*/results_run_*_model*.csv",
            # "*_150_interaction_true_norm*/results_run_*_model*.csv",
            "*_150_interaction_false_off*/results_run_*_model*.csv",
            # "*_150_interaction_true_off*/results_run_*_model*.csv",

            "*_300_interaction_false_norm*/results_run_*_model*.csv",
            # "*_300_interaction_true_norm*/results_run_*_model*.csv",
            "*_300_interaction_false_off*/results_run_*_model*.csv",
            # "*_300_interaction_true_off*/results_run_*_model*.csv"
        ]
        titles = ['Scenario 1', 'Scenario 3', 'Scenario 5', 'Scenario 7', 'Scenario 9', 'Scenario 11', 'Scenario 13',
                  'Scenario 15']
    else:
        scenarios = [
            "*_025_interaction_false_norm*/results_run_*_model*.csv",
            "*_025_interaction_true_norm*/results_run_*_model*.csv",
            "*_025_interaction_false_off*/results_run_*_model*.csv",
            "*_025_interaction_true_off*/results_run_*_model*.csv",

            "*_050_interaction_false_norm*/results_run_*_model*.csv",
            "*_050_interaction_true_norm*/results_run_*_model*.csv",
            "*_050_interaction_false_off*/results_run_*_model*.csv",
            "*_050_interaction_true_off*/results_run_*_model*.csv",

            "*_150_interaction_false_norm*/results_run_*_model*.csv",
            "*_150_interaction_true_norm*/results_run_*_model*.csv",
            "*_150_interaction_false_off*/results_run_*_model*.csv",
            "*_150_interaction_true_off*/results_run_*_model*.csv",

            "*_300_interaction_false_norm*/results_run_*_model*.csv",
            "*_300_interaction_true_norm*/results_run_*_model*.csv",
            "*_300_interaction_false_off*/results_run_*_model*.csv",
            "*_300_interaction_true_off*/results_run_*_model*.csv"
        ]
        titles = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6', 'Scenario 7',
                  'Scenario 8', 'Scenario 9', 'Scenario 10', 'Scenario 11', 'Scenario 12', 'Scenario 13', 'Scenario 14',
                  'Scenario 15', 'Scenario 16']

    scenario_data = {}

    for j, scenario in enumerate(scenarios):
        csv_files = list(directory.glob(scenario))

        df_results_total = []

        # Append all to one
        for i, csv_file in enumerate(csv_files):
            df_results = pd.read_csv(csv_file)

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df_results['timestamp'] = x_axis_time
            df_results.set_index('timestamp', inplace=True)
            # df_results['total_recharge_power'] = df_results['total_recharge_power']
            # df_results['total_load'] = df_results['total_recharge_power'] + df_results['total_customer_load']
            df_results_total.append(df_results)

        try:
            concatenated_df = pd.concat(df_results_total, ignore_index=False)
            concatenated_df['utilization'] = round(concatenated_df['total_load'], 2) / concatenated_df['transformer_capacity']
            print(concatenated_df['utilization'])
            # Define the bins
            bins = [0.0, 1.0, 1.25, 1.5, 2.0, float('inf')]
            labels = ['x <= 100%', '100% < x <= 125%', '125% < x <= 150%', '150% < x <= 200%', 'x > 200%']
            # Use pd.cut() to categorize transformer load percentages into bins
            concatenated_df['load_category'] = pd.cut(concatenated_df['utilization'], bins=bins, labels=labels, right=True)

            # Use value_counts() to count occurrences in each bin
            load_category_counts = concatenated_df['load_category'].value_counts()
            scenario_name = titles[j]
            scenario_data[scenario_name] = load_category_counts.to_dict()
        except:
            continue

    # Convert the dictionary to a DataFrame
    df_result = pd.DataFrame(scenario_data)
    # Transpose the DataFrame to have 'Load Category' as columns and 'Scenario' as the index
    df_result = df_result.transpose()

    # df_result = df_result[['100% - 150%', '150% - 200%', '200% - 250%', '250% - 300%', '300% - 350%', '350% - 400%', '>= 400%']]
    import seaborn as sns
    custom_palette = ["#008000", "#ffa500", "#ff0000", "#ff00ff", "#800080", "#336699", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_result.plot(kind='bar', stacked=True, edgecolor='black', width=0.7, linewidth=0.5, color=custom_palette, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("Quarters with Transformer Utilization All Runs")
    plt.xticks(rotation=90, ha='center')
    plt.legend(title="Transformer Utilization", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig("Transformer Utilization All Scenarios.png", dpi=300)
    plt.show()


def transformer_ueberlastung(split=False):
    """all | no | else"""
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    if split:
        scenarios = [
            "*_025_interaction_false_norm*/results_run_*_model*.csv",
            # "*_025_interaction_true_norm*/results_run_*_model*.csv",
            "*_025_interaction_false_off*/results_run_*_model*.csv",
            # "*_025_interaction_true_off*/results_run_*_model*.csv",

            "*_050_interaction_false_norm*/results_run_*_model*.csv",
            # "*_050_interaction_true_norm*/results_run_*_model*.csv",
            "*_050_interaction_false_off*/results_run_*_model*.csv",
            # "*_050_interaction_true_off*/results_run_*_model*.csv",

            "*_150_interaction_false_norm*/results_run_*_model*.csv",
            # "*_150_interaction_true_norm*/results_run_*_model*.csv",
            "*_150_interaction_false_off*/results_run_*_model*.csv",
            # "*_150_interaction_true_off*/results_run_*_model*.csv",

            "*_300_interaction_false_norm*/results_run_*_model*.csv",
            # "*_300_interaction_true_norm*/results_run_*_model*.csv",
            "*_300_interaction_false_off*/results_run_*_model*.csv",
            # "*_300_interaction_true_off*/results_run_*_model*.csv"
        ]
        titles = ['Scenario 1', 'Scenario 3', 'Scenario 5', 'Scenario 7', 'Scenario 9', 'Scenario 11', 'Scenario 13',
                  'Scenario 15']
    else:
        scenarios = [
            "*_025_interaction_false_norm*/results_run_*_model*.csv",
            "*_025_interaction_true_norm*/results_run_*_model*.csv",
            "*_025_interaction_false_off*/results_run_*_model*.csv",
            "*_025_interaction_true_off*/results_run_*_model*.csv",

            "*_050_interaction_false_norm*/results_run_*_model*.csv",
            "*_050_interaction_true_norm*/results_run_*_model*.csv",
            "*_050_interaction_false_off*/results_run_*_model*.csv",
            "*_050_interaction_true_off*/results_run_*_model*.csv",

            "*_150_interaction_false_norm*/results_run_*_model*.csv",
            "*_150_interaction_true_norm*/results_run_*_model*.csv",
            "*_150_interaction_false_off*/results_run_*_model*.csv",
            "*_150_interaction_true_off*/results_run_*_model*.csv",

            "*_300_interaction_false_norm*/results_run_*_model*.csv",
            "*_300_interaction_true_norm*/results_run_*_model*.csv",
            "*_300_interaction_false_off*/results_run_*_model*.csv",
            "*_300_interaction_true_off*/results_run_*_model*.csv"
        ]
        titles = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6', 'Scenario 7',
                  'Scenario 8', 'Scenario 9', 'Scenario 10', 'Scenario 11', 'Scenario 12', 'Scenario 13', 'Scenario 14',
                  'Scenario 15', 'Scenario 16']



    scenario_data = {}

    for j, scenario in enumerate(scenarios):

        csv_files = list(directory.glob(scenario))

        df_results_total = []

        # Append all to one
        for i, csv_file in enumerate(csv_files):
            df_results = pd.read_csv(csv_file)

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df_results['timestamp'] = x_axis_time
            df_results.set_index('timestamp', inplace=True)
            # df_results['total_recharge_power'] = df_results['total_recharge_power']
            # df_results['total_load'] = df_results['total_recharge_power'] + df_results['total_customer_load']
            df_results_total.append(df_results)

        try:
            concatenated_df = pd.concat(df_results_total, ignore_index=False)
            concatenated_df['utilization'] = round(concatenated_df['total_load'], 2) / concatenated_df[
                'transformer_capacity']

            # Define the bins
            bins = [0.0, 1.0, 1.25, 1.5, 2.0, float('inf')]
            labels = ['x <= 100%', '100% < x <= 125%', '125% < x <= 150%', '150% < x <= 200%', 'x > 200%']
            # Use pd.cut() to categorize transformer load percentages into bins
            concatenated_df['load_category'] = pd.cut(concatenated_df['utilization'], bins=bins, labels=labels,
                                                      right=True)

            # Use value_counts() to count occurrences in each bin
            load_category_counts = concatenated_df['load_category'].value_counts()
            scenario_name = titles[j]
            scenario_data[scenario_name] = load_category_counts.to_dict()
        except:
            continue

    # Convert the dictionary to a DataFrame
    df_result = pd.DataFrame(scenario_data)
    # Transpose the DataFrame to have 'Load Category' as columns and 'Scenario' as the index
    df_result = df_result.transpose()

    df_result = df_result[
        ['100% < x <= 125%', '125% < x <= 150%', '150% < x <= 200%', 'x > 200%']]

    import seaborn as sns
    custom_palette = ["#ffa500", "#ff0000", "#ff00ff", "#800080", "#336699", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_result.plot(kind='bar', stacked=True, edgecolor='black', width=0.7, linewidth=0.5, color=custom_palette, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("Quarters with Transformer Overloading All Runs")
    plt.xticks(rotation=90, ha='center')
    plt.legend(title="Transformer Overloading", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig('Transformer Absolute Overloading All Scenarios.png', dpi=300)
    plt.show()


def transformer_ueberlastung_prozent(split=False):
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    if split:
        scenarios = [
            "*_025_interaction_false_norm*/results_run_*_model*.csv",
            # "*_025_interaction_true_norm*/results_run_*_model*.csv",
            "*_025_interaction_false_off*/results_run_*_model*.csv",
            # "*_025_interaction_true_off*/results_run_*_model*.csv",

            "*_050_interaction_false_norm*/results_run_*_model*.csv",
            # "*_050_interaction_true_norm*/results_run_*_model*.csv",
            "*_050_interaction_false_off*/results_run_*_model*.csv",
            # "*_050_interaction_true_off*/results_run_*_model*.csv",

            "*_150_interaction_false_norm*/results_run_*_model*.csv",
            # "*_150_interaction_true_norm*/results_run_*_model*.csv",
            "*_150_interaction_false_off*/results_run_*_model*.csv",
            # "*_150_interaction_true_off*/results_run_*_model*.csv",

            "*_300_interaction_false_norm*/results_run_*_model*.csv",
            # "*_300_interaction_true_norm*/results_run_*_model*.csv",
            "*_300_interaction_false_off*/results_run_*_model*.csv",
            # "*_300_interaction_true_off*/results_run_*_model*.csv"
        ]
        titles = ['Scenario 1', 'Scenario 3', 'Scenario 5', 'Scenario 7', 'Scenario 9', 'Scenario 11', 'Scenario 13',
                  'Scenario 15']
    else:
        scenarios = [
            "*_025_interaction_false_norm*/results_run_*_model*.csv",
            "*_025_interaction_true_norm*/results_run_*_model*.csv",
            "*_025_interaction_false_off*/results_run_*_model*.csv",
            "*_025_interaction_true_off*/results_run_*_model*.csv",

            "*_050_interaction_false_norm*/results_run_*_model*.csv",
            "*_050_interaction_true_norm*/results_run_*_model*.csv",
            "*_050_interaction_false_off*/results_run_*_model*.csv",
            "*_050_interaction_true_off*/results_run_*_model*.csv",

            "*_150_interaction_false_norm*/results_run_*_model*.csv",
            "*_150_interaction_true_norm*/results_run_*_model*.csv",
            "*_150_interaction_false_off*/results_run_*_model*.csv",
            "*_150_interaction_true_off*/results_run_*_model*.csv",

            "*_300_interaction_false_norm*/results_run_*_model*.csv",
            "*_300_interaction_true_norm*/results_run_*_model*.csv",
            "*_300_interaction_false_off*/results_run_*_model*.csv",
            "*_300_interaction_true_off*/results_run_*_model*.csv"
        ]
        titles = ['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6', 'Scenario 7',
                  'Scenario 8', 'Scenario 9', 'Scenario 10', 'Scenario 11', 'Scenario 12', 'Scenario 13', 'Scenario 14',
                  'Scenario 15', 'Scenario 16']

    scenario_data = {}

    for j, scenario in enumerate(scenarios):
        csv_files = list(directory.glob(scenario))

        df_results_total = []

        # Append all to one
        for i, csv_file in enumerate(csv_files):
            df_results = pd.read_csv(csv_file)

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df_results['timestamp'] = x_axis_time
            df_results.set_index('timestamp', inplace=True)
            # df_results['total_recharge_power'] = df_results['total_recharge_power']
            # df_results['total_load'] = df_results['total_recharge_power'] + df_results['total_customer_load']
            df_results_total.append(df_results)

        try:
            concatenated_df = pd.concat(df_results_total, ignore_index=False)
            concatenated_df['utilization'] = round(concatenated_df['total_load'], 2) / concatenated_df[
                'transformer_capacity']

            # Define the bins
            bins = [0.0, 1.0, 1.25, 1.5, 2.0, float('inf')]
            labels = ['x <= 100%', '100% < x <= 125%', '125% < x <= 150%', '150% < x <= 200%', 'x > 200%']

            # categorize transformer load percentages into bins
            concatenated_df['load_category'] = pd.cut(concatenated_df['utilization'], bins=bins, labels=labels,
                                                      right=True)

            # Use value_counts() to count occurrences in each bin
            load_category_counts = concatenated_df['load_category'].value_counts()
            scenario_name = titles[j]
            scenario_data[scenario_name] = load_category_counts.to_dict()
        except:
            continue

    # Convert the dictionary to a DataFrame
    df_result = pd.DataFrame(scenario_data)
    # Transpose the DataFrame to have 'Load Category' as columns and 'Scenario' as the index
    df_result = df_result.transpose()

    df_result = df_result[
        ['100% < x <= 125%', '125% < x <= 150%', '150% < x <= 200%', 'x > 200%']]

    df_percent = df_result.div(df_result.sum(axis=1), axis=0) * 100
    import seaborn as sns

    custom_palette = ["#ffa500", "#ff0000", "#ff00ff", "#800080", "#336699", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(12, 6))
    df_percent.plot(kind='bar', stacked=True, edgecolor='black', width=0.7, linewidth=0.5, color=custom_palette, ax=ax)
    ax.set_ylabel("Percentage of Overloading\n[%]")
    ax.set_xlabel("")
    ax.set_title("Percentage of Transformer Overloading All Runs")
    plt.xticks(rotation=90, ha='center')
    plt.legend(title="Transformer Overloading", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig('Transformer Overloading all Scenarios.png', dpi=300)
    plt.show()

def box_plot_transformatorauslastung():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)

    results_dict = {}

    positions = []
    x_tick_labels = []
    fig, ax = plt.subplots(figsize=(12, 6))

    for j in range(1, 17):
        string = scenario_string(j)
        csv_files = list(directory.glob(string))

        df_results_total = []

        # Append all to one
        for i, csv_file in enumerate(csv_files):
            df_results = pd.read_csv(csv_file)

            x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
            x_axis_time = x_axis_time[:-1]

            df_results['timestamp'] = x_axis_time
            df_results.set_index('timestamp', inplace=True)
            # df_results['total_recharge_power'] = df_results['total_recharge_power']
            # df_results['total_load'] = df_results['total_recharge_power'] + df_results['total_customer_load']
            df_results_total.append(df_results)

        try:
            position = j
            positions.append(position)

            concatenated_df = pd.concat(df_results_total, ignore_index=False)
            concatenated_df['utilization'] = round(concatenated_df['total_load'], 2) / concatenated_df[
                'transformer_capacity'] * 100

            # print(concatenated_df['utilization'].std())
            scenario_name = 'Scenario {}'.format(j)

            x_tick_labels.append(scenario_name)
            ax.boxplot(concatenated_df['utilization'], patch_artist=True, positions=[position])#, showfliers=False)

            median_values = concatenated_df.median()
            mean_values = concatenated_df.mean()

            # Store the outcomes in the dictionary
            results_dict[scenario_name] = {
                'median': median_values,
                'mean': mean_values
            }

        except:
            continue
            # scenario_data[scenario_name] = load_category_counts.to_dict()

    plt.title('Transformer Utilization Boxplots')
    plt.ylabel('Transformer Utilization \n [%]')
    ax.set_xticklabels(x_tick_labels, rotation=90, ha='center')
    plt.tight_layout()
    plt.savefig('Boxplots_Transformer_Utilization.png', dpi=300)
    plt.show()
    plt.clf()
    # results_dict_df = pd.DataFrame.from_dict(results_dict, orient='index')
    # results_dict_df = results_dict_df.T
    # Create DataFrames for 'median' and 'mean'
    df_median = pd.DataFrame(results_dict).T['median'].apply(pd.Series)
    df_mean = pd.DataFrame(results_dict).T['mean'].apply(pd.Series)

    # Rename columns for 'median' and 'mean'
    df_median.columns = ['median_' + col for col in df_median.columns]
    df_mean.columns = ['mean_' + col for col in df_mean.columns]

    # Combine 'median' and 'mean' DataFrames
    df_results = pd.concat([df_median, df_mean], axis=1)
    # Define the desired order of columns
    desired_column_order = [
        "mean_total_customer_load",
        "median_total_customer_load",
        "mean_total_recharge_power",
        "median_total_recharge_power",
        "mean_total_load",
        "median_total_load",
        "mean_transformer_capacity",
        "median_transformer_capacity",
        "mean_utilization",
        "median_utilization",
    ]

    new_row_values = ['[kW]', '[kW]', '[kW]', '[kW]', '[kW]', '[kW]', '[kW]', '[kW]', '[%]', '[%]', ]
    new_row = pd.Series(new_row_values, index=desired_column_order)
    # Add the new row using loc
    df_results = df_results.round(2)

    def add_trailing_zero(val):
        try:
            return '{:.2f}'.format(float(val))
        except ValueError:
            return val

    # Apply the function to each cell of the DataFrame
    df_results = df_results.applymap(add_trailing_zero)

    # df_results = df_results.to_string(float_format='%.2f')
    df_results = pd.concat([pd.DataFrame([new_row], index=['unit']), df_results])
    df_results = df_results[desired_column_order]
    df_results = df_results.rename(columns={'mean_utilization': 'mean_transformer_utilization', 'median_utilization': 'median_transformer_utilization'})

    df_results = df_results.rename(columns=lambda x: x.replace('_', ' ').title())
    # Plot the DataFrame as a table
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.axis('off')  # Turn off axis

    # Use the pandas DataFrame.plot() function with kind='table' to plot the table
    table = ax.table(cellText=df_results.values, colLabels=df_results.columns, rowLabels=df_results.index, cellLoc='center', loc='center')

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


def spitzenlast_pro_auto():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    csv_files = list(directory.glob("model_30_*_interaction_false_*/results_all_runs.csv"))
    csv_files = sorted(csv_files)
    print(csv_files)

    data = []

    for i, csv_file in enumerate(csv_files):
        # Read the CSV file into a DataFrame

        df_results = pd.read_csv(csv_file)

        x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
        x_axis_time = x_axis_time[:-1]

        df_results['timestamp'] = x_axis_time
        df_results.set_index('timestamp', inplace=True)

        name = str(csv_file)
        if '025' in name:
            cars = 25
            spitzenlast_pro_auto = df_results['total_recharge_power'].max() / cars

        elif '050' in name:
            cars = 50
            spitzenlast_pro_auto = df_results['total_recharge_power'].max() / cars
        elif '150' in name:
            cars = 150
            spitzenlast_pro_auto = df_results['total_recharge_power'].max() / cars
        elif '300' in name:
            cars = 300
            spitzenlast_pro_auto = df_results['total_recharge_power'].max() / cars
        else:
            cars = None
            print("None")

        data.append((cars, spitzenlast_pro_auto))

    df = pd.DataFrame(data, columns=['Num Cars', 'Maximum Load'])
    df = df.explode('Maximum Load')
    plt.scatter(df['Num Cars'], df['Maximum Load'])
    plt.show()

if __name__ == '__main__':
    # create_diff_plots_all_runs_no_interaction()    # TODO
    # create_diff_plots_all_runs_interaction()       # TODO
    # average_load_profiles_models_one_agent()     # TODO Run this to create one plot with average load profiles for one agent
    # combination_of_average_load_profiles()
    # print_model_run_one_plot()   # TODO Run this to create average plot for each scenario // WRONG COLORS
    # create_heat_map()  # TODO Create Heatmap
    # spitzenlast_pro_auto()
    # plot_all()      # Run this to have comparison for different scenarios one fleet size
    prozentuale_ueberlastung(split=True)   # TODO CREATE PLOT
    transformer_ueberlastung(split=True)   # TODO CREATE PLOT
    transformer_ueberlastung_prozent(split=True)  # TODO CREATE PLOT
    # box_plot_transformatorauslastung()  # TODO Create Boxplot and copy table
    # all_model_results_and_average()    # TODO run this to create average results for missing scenario