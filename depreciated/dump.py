import mesa
from mesa import Model
from mesa.time import RandomActivation
from grid_agent import ElectricityGridBus
import pandas as pd
import datetime


def one_customer_base_load(start_date):
    # file = "h0_profile.csv"
    df = pd.read_csv(r"W:\abm_electric_vehicles\h0_profile.csv")
    df = df.drop(columns=['TagNr.', 'Tag'])

    # stack the rows and set the column name as index
    df_stacked = df.set_index('Datum').stack().reset_index(name='value').rename(columns={'level_1': 'time'})
    # combine the date and time columns into one datetime column
    df_stacked['datetime'] = pd.to_datetime(df_stacked['Datum'] + ' ' + df_stacked['time'],
                                            format='%d.%m.%Y %H:%M') - datetime.timedelta(minutes=15)
    # drop the original date and time columns
    df_stacked.drop(['Datum', 'time'], axis=1, inplace=True)
    # replace the year in h0 profile timestamps to current year
    relevant_year = pd.Timestamp(start_date).year
    df_stacked['datetime'] = df_stacked['datetime'].apply(lambda x: x.replace(year=relevant_year))
    # set the datetime column as index
    df_stacked.set_index('datetime', inplace=True)
    print(df_stacked)
    return df_stacked


# self.adjust_charging_values()
# # Get all Agents in buffer
# all_agents = self.model.schedule.agent_buffer(self)
# car_agents = [agent for agent in all_agents if isinstance(agent, ElectricVehicle)]
# car_agents_charging_values = [agent.charging_value for agent in car_agents]
# car_agents_charging_values_total = sum([x for x in car_agents_charging_values if x is not None])
#
# max_capacity = 25
#
# num_charging_cars = len([value for value in car_agents_charging_values if value is not None and value != 0])
#
# if car_agents_charging_values_total > max_capacity:
#     print("interaction")
#     exceeding_charging_value = car_agents_charging_values_total - max_capacity
#     reduction_per_agent = exceeding_charging_value / num_charging_cars
#
#     for agent in car_agents:
#         agent.charging_value = max(0, agent.charging_value - reduction_per_agent)
#     self.calculate_battery_level(reduced=True)
# self.calculate_battery_level(reduced=True)

# print("Agent with Car ID {}: current charging_value is: {}".format(self.car_id, self.charging_value))




if __name__ == '__main__':
    one_customer_base_load('2008-07-13')