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


# def set_plug_in_buffer(self, value: bool):
#     self.plug_in_buffer = value
#
# def set_plugged_in(self, value: bool):
#     self.plugged_in = value
#
# def get_plugged_in(self):
#     return self.plugged_in

# def set_plug_in_status(self):
#     """Function to check the plug in buffer and consumption to set the right plug in status."""
#     consumption = self.get_consumption()
#     plug_in_buffer = self.get_plug_in_buffer()
#     if consumption == 0 and plug_in_buffer == True:
#         self.set_plug_in_buffer(False)
#         self.set_plugged_in(False)
#     if consumption == 0 and plug_in_buffer == False:
#         self.set_plugged_in(True)
#     if consumption > 0:
#         self.set_plugged_in(False)
#         self.set_plug_in_buffer(True)

# def set_soc(self):
#     battery_lvl = self.get_battery_lvl()
#     battery_capacity = self.get_battey_capacity()
#     self.soc = battery_lvl / battery_capacity
#
# def get_soc(self):
#     return self.soc

# def set_target_soc_reached(self, value: bool):
#     self.target_soc_reached = value

# def set_target_soc_reached(self, target_soc):
#     soc = self.get_soc()
#     if soc >= target_soc:
#         self.set_target_soc_reached(True)
#     else:
#         self.set_target_soc_reached(False)

# def get_target_soc_reached(self):
#     return self.target_soc_reached

# def set_charging_value(self, value: float):
#     self.charging_value = value

# def main_charging_value(self):
#     plugged_in = self.get_plugged_in()
#     target_soc_reached = self.get_target_soc_reached()
#     if plugged_in == True and target_soc_reached == False:
#         value = calc_charging_value()
#         self.set_charging_value(value)
#     if plugged_in == False:
#         self.set_charging_value(0)
#     if plugged_in == True and target_soc_reached == True:
#         self.set_charging_value(0)
#
# def calc_charging_value():
#     a = empty_battery_capacity()
#     b = empty_battery_capacity_soc()
#     c = charging_power
#     # TODO MAYBE IMPLEMENT HERE CHARGING POWER OF CHARGER TOO
#     possible_charging_value = min(empty_battery_capacity, possible_soc_capacity, charging_power)
#     return possible_charging_value
#
# def empty_battery_capacity():
#     return battery_capacity - battery_lvl
#
# def empty_battery_capacity_soc():
#     potential_soc = target_soc - current_soc
#     possible_charging_value = battery_capacity * potential_soc
#     return possible_charging_value

# def get_right_charging_power_car():
#     """Can only charge at home or work."""
#     if cluster == 1:  # home
#         get_power_ac
#     elif cluster == 2:  # work
#         get_power_dc
#     else:
#         charging_power = 0

def get_right_charging_power_station():
    # TODO Maybe implement these charging values, but with what logic?
    chose = [3.7, 7.2, 11, 22]


if __name__ == '__main__':
    one_customer_base_load('2008-07-13')