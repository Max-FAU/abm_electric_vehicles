import datetime
import timeit
import pandas as pd
from model.model import ChargingModel
from tqdm import tqdm
from project_paths import RESULT_PATH
import numpy as np


if __name__ == '__main__':
    start = timeit.default_timer()
    start_date = '2008-07-13'
    end_date = '2008-07-27'
    model_runs = 1

    num_cars_normal = 100
    num_cars_off_peak = 0
    num_transformers = 1
    num_customers = 100
    car_target_soc = 100
    car_charging_algo = False

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    # model_results = []
    # Run the whole model multiple times
    for i in tqdm(range(model_runs), desc='Model Runs', leave=True, position=0):
        # Create each iteration of the model a new seed value
        seed_value = np.random.randint(low=0, high=99999)

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
        # model_results.append(model_data)

        agent_data = model.datacollector.get_agent_vars_dataframe()
        agent_data.to_csv(RESULT_PATH / "results_run_{}_agent_data.csv".format(i), index=False)

        with open(RESULT_PATH / "seed_run_{}.txt".format(i), "w") as file:
            file.write(str(seed_value))

    # df_concat = pd.concat(model_results)
    # df_results = df_concat.groupby(df_concat.index).mean()
    # df_results.to_csv(RESULT_PATH / "results_all_runs.csv", index=False)

    end = timeit.default_timer()
    run_time = end - start
    print(f"Total run time: {run_time / 60} minutes")
