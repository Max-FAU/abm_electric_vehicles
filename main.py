import pandas as pd
import mobility_data as md


if __name__ == '__main__':
    path = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    mobility_data = pd.read_csv(path)
    mobility_data = md.prepare_mobility_data(df=mobility_data,
                                          starting_date='2008-07-12 00:00:00',
                                          days=1)

    mobility_data_aggregated = md.aggregate_15_min_steps(mobility_data)
    print(mobility_data_aggregated)