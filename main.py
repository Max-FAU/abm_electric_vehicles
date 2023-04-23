import datetime
import pandas as pd
from model import ChargingModel

if __name__ == '__main__':
    start_date = '2008-07-13'
    end_date = '2008-07-14'
    num_agents = 1

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model = ChargingModel(num_agents, start_date, end_date)

    for i in range(num_intervals):
        model.step()
