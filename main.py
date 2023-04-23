import datetime
import timeit
import pandas as pd
from model import ChargingModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = timeit.default_timer()
    start_date = '2008-07-13'
    end_date = '2008-07-20'
    num_agents = 10

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))
    # num_intervals = 24
    model = ChargingModel(num_agents, start_date, end_date)

    for i in range(num_intervals):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()
    model_data['total_recharge_value'].plot(linewidth=0.5)
    plt.show()

    agent_data = model.datacollector.get_agent_vars_dataframe()
    agent_ids = list(agent_data.index.get_level_values('AgentID').unique())
    mask = agent_data.index.get_level_values('AgentID').isin(agent_ids)
    agents_data = agent_data[mask]
    # Plot the battery_level column for each selected agent on the same plot
    agents_data['recharge_value'].unstack().plot(linewidth=0.5)
    plt.show()

    stop = timeit.default_timer()
    print('Total Runtime: ', stop - start)