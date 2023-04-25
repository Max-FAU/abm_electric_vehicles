import datetime
import math
import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
import pandas as pd
import matplotlib.pyplot as plt

import grid_agent


class TestEV(Agent):
    def __int__(self, model, unique_id, charging_power):
        super().__init__(model, unique_id)
        self.charging_power = charging_power

    def step(self):
        pass


class ElectricityGridBus(Agent):
    def __init__(self,
                 model,
                 unique_id,
                 num_households,
                 yearly_cons_household,
                 start_date: str,
                 end_date: str):
        super().__init__(unique_id, model)
        # self.unique_id = unique_id
        self.start_date = start_date
        self.end_date = end_date
        self.num_households = num_households
        self.volt_house_hold = 230
        self.ampere_house_hold = 40
        self.phases = 3
        self.power_house_hold = self.volt_house_hold * self.ampere_house_hold * self.phases

        self.customers_contracted_power = []
        self.get_customers_contracted_power()

        self.f_safety = 1.5
        self.p_over = 10
        self.transformer_capacity = self.calc_transformer_power_capacity()

        self.yearly_cons_household = yearly_cons_household
        self.scale = self.yearly_cons_household / 1000
        self.standard_load_profile = self.one_customer_base_load()
        self.scaled_load_profile = self.scale_one_customer_base_load()

        self.total_base_load = self.set_total_base_load()
        #TODO start_date and end_date, create a dataframe with 15 min steps,

        self.current_load = None   # from base load
        self.current_timestamp = None

    def set_timestamp(self, timestamp):
        self.current_timestamp = timestamp

    def set_current_load(self):
        """Base load for the corresponding timestamp."""
        self.current_load = self.total_base_load.loc[self.current_timestamp, 'value']

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
        relevant_year = pd.Timestamp(self.start_date).year
        df_stacked['datetime'] = df_stacked['datetime'].apply(lambda x: x.replace(year=relevant_year))
        # set the datetime column as index
        df_stacked.set_index('datetime', inplace=True)

        return df_stacked

    def scale_one_customer_base_load(self):
        return self.standard_load_profile * self.scale

    def set_total_base_load(self):
        """For number of households."""
        return self.scaled_load_profile * self.num_households

    def step(self):
        if self.current_timestamp is None:
            self.current_timestamp = self.start_date
            self.current_timestamp = pd.to_datetime(self.current_timestamp)
        else:
            # each step add 15 minutes
            self.current_timestamp = self.current_timestamp + datetime.timedelta(minutes=15)

    # def unnessesary(self):
    #     charging_dict = {
    #         'low': 3.7,
    #         'normal': 11,
    #         'high': 22
    #     }


class StartModel(Model):
    def __init__(self, num_agents: int, num_households: int):
        self.num_agents = 1
        self.num_households = num_households
        self.num_agents = num_agents
        self.num_transformers = num_agents // num_households + 1
        self.max_transformer_capacity = 25  # kW

        self.power_reduction = None

        self.schedule = RandomActivation(self)

        for i in range(self.num_transformers):
            transformer = ElectricityGridBus(model=self,
                                             unique_id=i,
                                             num_households=num_households,
                                             yearly_cons_household=3500)
            self.schedule.add(transformer)

        car_model = ['renault_zoe']
        for j in range(self.num_agents):
            agent = TestEV(unique_id=i,
                            car_model=car_model,
                            target_soc=1.0,
                            start_date=self.start_date,
                            end_date=self.end_date,
                            model=self)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()


if __name__ == '__main__':

    model = StartModel(num_agents=1, num_households=24)

    for j in range(1):
        model.step()

    # # We could take one of these transformers, e.g. ABB DRY-TYPE TRANSFORMER 25 kVA 480-120/240V
    # # https://electrification.us.abb.com/products/transformers-low-voltage-dry-type
    # # take siemens https://mall.industry.siemens.com/mall/de/WW/Catalog/Products/10283675
    # # size depends on the phases we want
    # # usually we have as output 400 V
    # num_households = 24
    # transformer = ElectricityGridBus(1, num_households, 3500)
    # print(transformer.transformer_capacity)
    # # print(transformer.customers_contracted_power)
    # df_stacked = transformer.total_base_load / 1000
    # # print(transformer.scaled_load_profile)
    # # print(sum(transformer.standard_load_profile['value']) / 1000 / 4) # kwh
    #
    # sum = sum(transformer.total_base_load['value']) / 4 / 1000
    #
    # print(sum)
    # df_stacked.plot()
    # plt.show()
