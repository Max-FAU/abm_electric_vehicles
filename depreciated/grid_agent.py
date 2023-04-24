import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt


class ElectricityGridBus:
    def __init__(self, unique_id, num_households, yearly_cons_household):
        self.unique_id = unique_id
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
        file = r"C:\Users\Max\PycharmProjects\mesa\h0_profile.csv"
        df = pd.read_csv(file)
        df = df.drop(columns=['TagNr.', 'Tag'])

        # stack the rows and set the column name as index
        df_stacked = df.set_index('Datum').stack().reset_index(name='value').rename(columns={'level_1': 'time'})
        # combine the date and time columns into one datetime column
        df_stacked['datetime'] = pd.to_datetime(df_stacked['Datum'] + ' ' + df_stacked['time'], format='%d.%m.%Y %H:%M') - datetime.timedelta(minutes=15)
        # drop the original date and time columns
        df_stacked.drop(['Datum', 'time'], axis=1, inplace=True)
        # set the datetime column as index
        df_stacked.set_index('datetime', inplace=True)
        return df_stacked

    def scale_one_customer_base_load(self):
        return self.standard_load_profile * self.scale

    def set_total_base_load(self):
        return self.scaled_load_profile * self.num_households

    def unnessesary(self):
        charging_dict = {
            'low': 3.7,
            'normal': 11,
            'high': 22
        }


if __name__ == '__main__':
    num_households = 24
    transformer = ElectricityGridBus(1, num_households, 3500)
    # print(transformer.transformer_capacity)
    # print(transformer.customers_contracted_power)
    df_stacked = transformer.total_base_load / 1000
    # print(transformer.scaled_load_profile)
    # print(sum(transformer.standard_load_profile['value']) / 1000 / 4) # kwh

    sum = sum(transformer.total_base_load['value']) / 4 / 1000

    print(sum)
    df_stacked.plot()
    plt.show()