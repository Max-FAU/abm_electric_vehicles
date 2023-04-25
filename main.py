import datetime
import timeit
import pandas as pd
from model import ChargingModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = timeit.default_timer()
    start_date = '2008-07-13'
    end_date = '2008-07-27'
    num_agents = 698
    model_runs = 30

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model_results = []
    # Run the whole model x times
    for i in range(model_runs):
        print("Start iteration {} of model runs...".format(i + 1))
        model = ChargingModel(num_agents, start_date, end_date)

        for j in range(num_intervals):
            model.step()

        model_data = model.datacollector.get_model_vars_dataframe()
        model_data.to_csv("results/results_model_run_{}.csv".format(i), index=False)
        model_results.append(model_data)

        print("...finished iteration {} of model runs.".format(i + 1))

    df_concat = pd.concat(model_results)
    df_results = df_concat.groupby(df_concat.index).mean()
    df_results.to_csv("results/results_all_runs.csv", index=False)

    df_results['total_recharge_value'].plot(linewidth=0.5)
    plt.show()
    #
    # agent_data = model.datacollector.get_agent_vars_dataframe()
    # agent_ids = list(agent_data.index.get_level_values('AgentID').unique())
    # mask = agent_data.index.get_level_values('AgentID').isin(agent_ids)
    # agents_data = agent_data[mask]
    # # Plot the battery_level column for each selected agent on the same plot
    # agents_data['recharge_value'].unstack().plot(linewidth=0.5, legend=None)
    # plt.show()
    #
    # stop = timeit.default_timer()
    # print('Total Runtime: ', stop - start)