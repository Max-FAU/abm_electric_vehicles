import datetime
import timeit
import pandas as pd
from model.model import ChargingModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from project_paths import RESULT_PATH


if __name__ == '__main__':
    start = timeit.default_timer()
    start_date = '2008-07-13'
    end_date = '2008-07-20'
    num_cars = 2
    num_transformers = 1
    num_customers = 2
    model_runs = 2

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model_results = []
    # Run the whole model multiple times
    for i in range(model_runs):
        print("Start iteration {} of model runs...".format(i + 1))
        model = ChargingModel(num_cars,
                              num_transformers,
                              num_customers,
                              start_date,
                              end_date)

        for j in tqdm(range(num_intervals)):
            model.step()

        model_data = model.datacollector.get_model_vars_dataframe()
        model_data["total_load"] = model_data["total_recharge_power"] + model_data["total_customer_load"]
        model_data.to_csv(RESULT_PATH / "results_run_{}_model_data.csv".format(i), index=False)
        model_results.append(model_data)

        agent_data = model.datacollector.get_agent_vars_dataframe()
        agent_data.to_csv(RESULT_PATH / "results_run_{}_agent_data.csv".format(i), index=False)
        print("...finished iteration {} of model runs.".format(i + 1))

    df_concat = pd.concat(model_results)
    df_results = df_concat.groupby(df_concat.index).mean()
    df_results.to_csv(RESULT_PATH / "results_all_runs.csv", index=False)

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

    plt.subplots_adjust(bottom=0.3)

    plt.tight_layout()
    plt.show()
    end = timeit.default_timer()
    run_time = end - start
    print(f"Total run time: {run_time} seconds")
