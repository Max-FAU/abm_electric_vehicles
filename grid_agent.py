import datetime
import math
from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler, RandomActivationByType, SimultaneousActivation
import pandas as pd
import matplotlib.pyplot as plt
from car_agent import ElectricVehicle
import numpy as np
import json
from mesa.datacollection import DataCollector
from own_scheduler import DualStepScheduler


# class ElectricityGridBus(Agent):
class ElectricityGridBus:
    def __init__(self,
                 # model,
                 unique_id,
                 num_households,
                 yearly_cons_household,
                 start_date: str,
                 end_date: str):

        # super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_households = num_households  # num_households == num EV Agents
        # TODO MAYBE REFACTOR EVERYTHING HERE IN ONE FUNCTION SINCE VALUES NOT CHANGING
        self.volt_house_hold = 230  # Volt
        self.ampere_house_hold = 40
        self.phases = 3  # maybe 1
        self.power_house_hold = self.volt_house_hold * self.ampere_house_hold * self.phases

        self.customers_contracted_power = []
        self.get_customers_contracted_power()

        self.f_safety = 1.5
        self.p_over = 10
        self.transformer_capacity = self.calc_transformer_power_capacity()

        self.transformer_capacity_test = 25  # kw

        self.yearly_cons_household = yearly_cons_household
        self.scale = self.yearly_cons_household / 1000
        self.standard_load_profile = self.one_customer_base_load()  # 1000 kwh yearly
        # print(sum(self.standard_load_profile['value']) / 4 / 1000)
        self.scaled_load_profile = self.scale_one_customer_base_load()  # * 3.5

        self.total_base_load = self.set_total_base_load()  # * 24

        self.current_load = None  # from base load in W
        self.current_load_kw = None  # in kW
        self.current_timestamp = None

        self.current_charging_power_total = 0

    def get_test_transformer_capacity(self):
        return self.transformer_capacity_test

    def set_timestamp(self, timestamp):
        self.current_timestamp = timestamp

    def set_current_load(self):
        """Base load for the corresponding timestamp."""
        self.current_load = self.total_base_load.loc[self.current_timestamp, 'value']

    def set_current_load_kw(self):
        self.current_load_kw = round(self.current_load / 1000, 1)

    def get_customers_contracted_power(self):
        for i in range(self.num_households):
            # in kW
            single_customer_value = self.power_house_hold / 1000
            self.customers_contracted_power.append(single_customer_value)

    def get_c_diversity(self):
        return 0.2 + 0.8 / math.sqrt(self.num_households)

    def calc_transformer_power_capacity(self):
        """
        p_pt = transformer power capacity
        c_diversity = 0.2 + 0.8 / sqrt(n)   # this is only valid for private customers
        c_diverstiy = 0.5 + 0.5 / sqrt(n)
        n = expected number of customers
        f_safety = safety margin to account for the power factor and for future load growth
        c_diversity = diversity (or simultaneity) coefficient
        p_over = oversized power capacity defined by the standard size of the transformer
        https://www.sciencedirect.com/science/article/pii/S0960148117310649?via%3Dihub

        round it to the next 100: 370 -> 400
        """
        return round(self.get_c_diversity() * sum(self.customers_contracted_power) * self.f_safety + self.p_over, -2)

    def one_customer_base_load(self):
        # file = "h0_profile.csv"
        df = pd.read_csv("h0_profile.csv")
        # df = pd.read_csv(r"W:\abm_electric_vehicles\h0_profile.csv")
        df = df.drop(columns=['TagNr.', 'Tag'])

        # stack the rows and set the column name as index
        df_stacked = df.set_index('Datum').stack().reset_index(name='value').rename(columns={'level_1': 'time'})
        # combine the date and time columns into one datetime column
        df_stacked['datetime'] = pd.to_datetime(df_stacked['Datum'] + ' ' + df_stacked['time'],
                                                format='%d.%m.%Y %H:%M') - datetime.timedelta(minutes=15)
        # drop the original date and time columns
        df_stacked.drop(['Datum', 'time'], axis=1, inplace=True)
        # replace the year in h0 profile timestamps to current year
        relevant_year = self.start_date.year
        # print(relevant_year)
        df_stacked['datetime'] = df_stacked['datetime'].apply(lambda x: x.replace(year=relevant_year))
        # set the datetime column as index
        df_stacked.set_index('datetime', inplace=True)
        return df_stacked

    def scale_one_customer_base_load(self):
        return self.standard_load_profile * self.scale

    def set_total_base_load(self):
        """For number of households."""
        return self.scaled_load_profile * self.num_households

    def set_timestamp(self):
        if self.current_timestamp is None:
            self.current_timestamp = self.start_date
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
        else:
            # each step add 15 minutes
            self.current_timestamp = self.current_timestamp + datetime.timedelta(minutes=15)

    def step(self):
        self.set_timestamp()
        self.set_current_load()
        self.set_current_load_kw()

        # print("Transformer with a capacity of {} kW".format(self.transformer_capacity_test))

    # def unnessesary(self):
    #     charging_dict = {
    #         'low': 3.7,
    #         'normal': 11,
    #         'high': 22
    #     }


class StartModel(Model):
    def __init__(self, num_agents: int,
                 num_households_per_transformer: int,
                 start_date: str,
                 end_date: str):

        self.start_date = start_date
        self.end_date = end_date
        self.num_households_per_transformer = num_households_per_transformer

        self.num_agents = num_agents  # agents are number of EV Agents
        self.num_transformers = num_agents // num_households_per_transformer + 1

        print("We generate {} cars.".format(self.num_agents))
        print("We have {} transformers for them.".format(self.num_transformers))

        # self.power_reduction = None
        # self.schedule = RandomActivationByType(self)
        self.schedule = SimultaneousActivation(self)

        self.datacollector = DataCollector(
            model_reporters={"total_charging_power": lambda m: sum(
                a.charging_value for a in m.schedule.agents if isinstance(a, ElectricVehicle)),
                             "test_transformer_capacity": lambda t: sum(trans.transformer_capacity_test for trans in
                                                                        t.schedule.agents if
                                                                        isinstance(trans, ElectricityGridBus)) /
                                                                    len([trans.transformer_capacity_test for trans in
                                                                         t.schedule.agents if
                                                                         isinstance(trans, ElectricityGridBus)])}
            # calculate the average here
        )

        for i in range(self.num_transformers):
            transformer = ElectricityGridBus(model=self,
                                             unique_id=i,
                                             num_households=num_households_per_transformer,
                                             yearly_cons_household=3500,
                                             start_date=self.start_date,
                                             end_date=self.end_date)
            self.schedule.add(transformer)

        self.maximum_transformer_capacity = transformer.transformer_capacity_test

        self.list_models = self.generate_cars_according_to_dist()

        # use k because i has already been taken
        k = 0
        while k < len(self.list_models):
            car_model = self.list_models[k]
            try:
                agent = ElectricVehicle(unique_id=k + 1000,  # add 1000 to have unique ids
                                        car_model=car_model,
                                        target_soc=1.0,
                                        start_date=self.start_date,
                                        end_date=self.end_date,
                                        model=self)
                self.schedule.add(agent)
            except Exception as e:
                print("Adding agent to model failed.")

            print("Added agent number {} to the model.".format(k))
            k += 1

    def generate_cars_according_to_dist(self):
        with open('car_values.json', 'r') as f:
            data = json.load(f)

        total_cars = 0
        for name in data.keys():
            total_cars += data[name]["number"]

        cars = []
        distribution = []
        for name in data.keys():
            cars += [name]
            distribution += [data[name]["number"] / total_cars]

        car_models = np.random.choice(cars, size=self.num_agents, p=distribution)
        # print(len(car_names), "car names generated.")

        return car_models

    # def calc_new_charging_per_car(self, agent):
    #     current_charging_value = agent.get_current_charging_value()

    def step(self):
        # if self.schedule.steps > 0:
        #     self.datacollector.collect(self)
        # step through schedule
        self.schedule.step()
        self.datacollector.collect(self)

        charging_data = self.datacollector.get_model_vars_dataframe()
        # total_charging_value = charging_data.loc[self.schedule.steps - 1, 'total_charging_power']

        # print(charging_data)
        # total_charging_value = charging_data.loc[self.schedule.steps, 'total_charging_power']
        #
        # charging_agents = []
        # for agent in self.schedule.agents:
        #     if agent.charging_value > 0:
        #         charging_agents.append(agent)
        #
        # max_capacity = 25
        #
        # while total_charging_value > max_capacity:
        #     exceeding_charging_value = total_charging_value - max_capacity
        #     reduction_per_agent = exceeding_charging_value / len(charging_agents)
        #     for agent in charging_agents:
        #         agent.charging_value = max(0, agent.charging_value - reduction_per_agent)
        #     charging_agents = [agent for agent in self.schedule.agents if agent.charging_value > 0]
        #     total_charging_value = sum(agent.charging_value for agent in charging_agents)

        # total_charging_value = test.loc[self.schedule.steps - 1, 'total_charging_power']
        # transformer_capacity = test.loc[self.schedule.steps - 1, 'test_transformer_capacity']

        # if total_charging_value > transformer_capacity:
        #     for a in self.schedule.agents:
        #         if isinstance(a, ElectricVehicle):
        #             a.charging_value = 0

        # TODO MAYBE IMPLEMENT A FUNCTION IN AGENT STEP TO RETRIEVE
        # TODO THE NEW CHARGING VALUE CALCULATED HERE

        if self.schedule.steps == 96:
            test = self.datacollector.get_model_vars_dataframe()
            test.plot()
            plt.show()


if __name__ == '__main__':

    start_date = '2008-07-13 14:15:00'
    end_date = '2008-07-14 15:30:00'

    # time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    # num_intervals = int(time_diff / datetime.timedelta(minutes=15))
    #
    # model = StartModel(num_agents=6,
    #                    num_households_per_transformer=24,
    #                    start_date=start_date,
    #                    end_date=end_date)
    #
    # for j in range(num_intervals):
    #     model.step()

    # # We could take one of these transformers, e.g. ABB DRY-TYPE TRANSFORMER 25 kVA 480-120/240V
    # # https://electrification.us.abb.com/products/transformers-low-voltage-dry-type
    # # take siemens https://mall.industry.siemens.com/mall/de/WW/Catalog/Products/10283675
    # # size depends on the phases we want
    # # usually we have as output 400 V
    #
    # print(sum)
    # df_stacked.plot()
    # plt.show()

    transformer = ElectricityGridBus(unique_id=0,
                                     num_households=24,
                                     yearly_cons_household=3500,
                                     start_date=start_date,
                                     end_date=end_date)
    transformer.set_timestamp()
    transformer.set_current_load()
    transformer.set_current_load_kw()
    print(transformer.power_house_hold)

    print(transformer.transformer_capacity)
    print(transformer.current_timestamp)
    print(transformer.current_load_kw)
