import pandas as pd
from depreciated.car_agent_old import ElectricVehicle

if __name__ == '__main__':
    # path = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
    path = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    raw_mobility_data = pd.read_csv(path)
    car_models = ['vw_golf']
    results = {}

    i = 0
    for car in car_models:
        individual_identifier = i
        generated_car = ElectricVehicle(model=car)
        generated_car.add_mobility_data(mobility_data=raw_mobility_data,
                                        starting_date='2008-07-13',
                                        num_days=1)
        # TODO fix generated_car.unique_id
        # print(generated_car.unique_id)
        timestamps = []

        for timestamp, data_row in generated_car.mobility_data.iterrows():
            battery_level = generated_car.calculate_battery_level(consumption=data_row['ECONSUMPTIONKWH'],
                                                                  battery_efficiency=100)
            timestamps.append(timestamp)

        outcome = pd.DataFrame(
            {
                'timestamp': timestamps,
                'battery_level': generated_car.battery_level_curve,
                'load_curve': generated_car.load_curve,
                'soc': generated_car.soc_curve,
                'id': generated_car.unique_id
            }
        ).set_index('timestamp')

        results[individual_identifier] = outcome
        i += 1

    # Aggregation by 15 Mins
    df_results_aggregated = pd.concat(results.values(), axis=0).groupby(pd.Grouper(freq='15Min')).mean()

    # Print the concatenated dataframe
    df_results_aggregated.to_csv('aggregated_results.csv')
