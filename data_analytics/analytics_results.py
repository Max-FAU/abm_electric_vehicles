from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FixedLocator, MultipleLocator, FixedFormatter, MaxNLocator, LogLocator


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
def create_diff_plots_all_runs():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    csv_files = list(directory.glob("**/results_all_runs.csv"))
    csv_files = sorted(csv_files)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 14))
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
        row = i // 4  # Calculate the row index
        col = i % 4  # Calculate the column index
        ax = axes[row, col]  # Select the subplot

        df_grouped.plot(y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'], ax=ax)
        # df_grouped.plot(y='total_recharge_power', ax=ax)
        title = csv_file.parent.name
        title = title.replace("model_30_", "").replace("_cars", "").replace("_interaction", "")
        ax.set_title(title)
        ax.set_ylabel('Charging Power \n [kW]')
        ax.legend().remove()

        ax.set_ylim(bottom=0, top=4)

        if row != 4:
            # Remove x-axis labels from all subplots except the last row
            ax.set_xticklabels([])

        if row == 4:
            # Set x-axis label for the last row of subplots
            ax.set_xlabel('Time [h]')

        if i == len(csv_files) - 1:
            # Store lines from the last subplot
            lines_last_subplot = ax.get_lines()



    # Create a single legend below the plot using lines from the last subplot
    legend = plt.figlegend(lines_last_subplot, [line.get_label() for line in lines_last_subplot],
                           loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=False)
    legend.get_frame().set_facecolor('white')
    plt.subplots_adjust(bottom=0.15, wspace=0.35, hspace=0.5)
    plt.show()


def create_same_plot_all_runs():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    csv_files = list(directory.glob("model_30_025*/results_all_runs.csv"))
    csv_files = sorted(csv_files)

    fig, ax = plt.subplots(figsize=(12, 6))
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

        df_grouped['time'] = df_grouped['hour'].astype(str) + ':' + df_grouped['minute'].astype(str)
        df_grouped.plot(x='time', y='total_recharge_power', ax=ax)
        title = csv_file.stem  # Use the filename as the title
        ax.set_title(title)
        ax.set_ylabel('Charging Power [kW]')
        ax.legend().remove()
        ax.set_xlim(0, 96)
        ax.set_ylim(0, 1.2)

        if i == len(csv_files) - 1:
            # Store lines from the last subplot
            lines_last_subplot = ax.get_lines()

    # Set a fixed number of y-axis tick locators
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Create a single legend below the plot using lines from the last subplot
    legend = plt.figlegend(lines_last_subplot, [line.get_label() for line in lines_last_subplot],
                           loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=False)
    legend.get_frame().set_facecolor('white')
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def print_scenario_one_row():
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    directory_path = r'C:\Users\Max\PycharmProjects\mesa\results'
    directory = Path(directory_path)
    csv_files = list(directory.glob("**/results_all_runs.csv"))
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

        x = df.index
        y = df['total_recharge_power']
        y1 = df['total_customer_load']
        y2 = df['total_load']
        y3 = df['transformer_capacity']

        plt.plot(x, y, label='total_recharge_power')
        plt.plot(x, y1, label='total_customer_load')
        plt.plot(x, y2, label='total_load')
        plt.plot(x, y3, label='transformer_capacity')


        # df.plot(y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'], ax=ax)
        # plt.xticks(x, rotation=90)

        import matplotlib.dates as mdates
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.xticks(rotation=90)

        title = "Scenario " + str(i + 1)
        plt.title(title)
        plt.ylabel('Charging Power \n [kW]')

        plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.6), frameon=False)
        plt.ylim(0, 6)
        plt.xlim(df.index.min(), df.index.max())
        # ax.set_xlim(0, 14)
        plt.xlabel('Day')

        # Add a legend
        # plt.legend()
        plt.tight_layout()
        fig_name = title + '_weekly_profile'
        plt.savefig(fig_name, dpi=300)
        plt.show()


if __name__ == '__main__':
    # create_diff_plots_all_runs()
    # create_same_plot_all_runs()
    print_scenario_one_row()