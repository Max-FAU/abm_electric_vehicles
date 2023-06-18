import datetime
import timeit
import pandas as pd
from model.model import ChargingModel
from tqdm import tqdm
from project_paths import RESULT_PATH
import numpy as np
import argparse

if __name__ == '__main__':
    start = timeit.default_timer()
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    parser = argparse.ArgumentParser(description='Run simulation with different parameters to generate load profiles.')
    parser.add_argument('--model_runs', type=int, default=1, help='Number of model runs')
    parser.add_argument('--num_cars_normal', type=int, default=2, help='Number of normal cars')
    parser.add_argument('--num_cars_off_peak', type=int, default=0, help='Number of off-peak cars')
    parser.add_argument('--num_transformers', type=int, default=1, help='Number of transformers')
    parser.add_argument('--num_customers', type=int, default=2, help='Number of customers')
    args = parser.parse_args()

    model_runs = args.model_runs
    num_cars_normal = args.num_cars_normal
    num_cars_off_peak = args.num_cars_off_peak
    num_transformers = args.num_transformers
    num_customers = args.num_customers

    car_charging_algo = True

    car_target_soc = 100

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model_results = []
    # Run the whole model multiple times
    for i in tqdm(range(model_runs), desc='Model Runs', leave=True, position=0):
        # Create each iteration of the model a new seed value
        seed_value = np.random.randint(low=0, high=9999)

        model = ChargingModel(num_cars_normal,
                              num_cars_off_peak,
                              num_transformers,
                              num_customers,
                              start_date,
                              end_date,
                              car_target_soc,
                              car_charging_algo,
                              seed_value)

        for j in tqdm(range(num_intervals), desc='Steps', leave=False, position=1):
            model.step()

        model_data = model.datacollector.get_model_vars_dataframe()
        model_data["total_load"] = model_data["total_recharge_power"] + model_data["total_customer_load"]
        model_data.to_csv(RESULT_PATH / "results_run_{}_model_data.csv".format(i), index=False)
        model_results.append(model_data)

        agent_data = model.datacollector.get_agent_vars_dataframe()
        agent_data.to_csv(RESULT_PATH / "results_run_{}_agent_data.csv".format(i), index=False)

        with open(RESULT_PATH / "seed_run_{}.txt".format(i), "w") as file:
            file.write(str(seed_value))

    df_concat = pd.concat(model_results)
    df_results = df_concat.groupby(df_concat.index).mean()
    df_results.to_csv(RESULT_PATH / "results_all_runs.csv", index=False)

    end = timeit.default_timer()
    run_time = end - start
    print(f"Total run time: {run_time / 60} minutes")
    #
    from depreciated.plot import create_plot
    create_plot(start_date, end_date, df_results)