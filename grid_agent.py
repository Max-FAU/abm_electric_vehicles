import datetime
import math
import pandas as pd


class PowerCustomer:
    def __init__(self, timestamp, yearly_cons_household, start_date: str, end_date: str):
        self.timestamp = timestamp
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.yearly_cons_household = yearly_cons_household

        self.scale = self.yearly_cons_household / 1000
        self.standard_load_profile = self.one_customer_base_load()  # 1000 kwh yearly
        # print(sum(self.standard_load_profile['value']) / 4 / 1000)
        self.scaled_load_profile = self.scale_customer_base_load()  # * 3.5
        # print(sum(self.scaled_load_profile['value']) / 1000 / 4 / 3500)
        self.current_load = None  # from base load in W
        self.current_load_kw = None  # in kW

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

    def scale_customer_base_load(self):
        return self.standard_load_profile * self.scale

    def set_current_load(self):
        """Base load for the corresponding timestamp."""
        self.current_load = self.scaled_load_profile.loc[self.timestamp, 'value']

    def set_current_load_kw(self):
        self.current_load_kw = self.current_load / 1000

    def get_current_load_kw(self):
        return self.current_load_kw


class Transformer:
    def __init__(self,
                 unique_id,
                 num_households):

        self.unique_id = unique_id
        self.num_households = num_households  # num_households == num EV Agents

        self.power_house_hold = self.set_power_house_hold()

        # calculation of transformer capacity
        self.customers_contracted_power = []
        self.get_customers_contracted_power()
        self.f_safety = 1.5
        self.p_over = 10
        self.transformer_capacity = self.calc_transformer_power_capacity()

    @staticmethod
    def set_power_house_hold():
        volt = 230
        ampere = 63
        phases = 1
        return volt * ampere * phases

    def get_customers_contracted_power(self):
        for i in range(self.num_households):
            single_customer_value = self.power_house_hold / 1000  # in kW
            self.customers_contracted_power.append(single_customer_value)

    def get_c_diversity(self):
        """Set diversity factor, used if load profiles created artificially."""
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

        """
        # print("Transformer with a capacity of {} kW".format(self.transformer_capacity))
        return self.get_c_diversity() * sum(self.customers_contracted_power) * self.f_safety + self.p_over


if __name__ == '__main__':

    interval = datetime.timedelta(minutes=15)
    start_date = pd.to_datetime('2008-07-13 00:00:00')
    end_date = pd.to_datetime('2008-07-13 23:00:00')

    # # We could take one of these transformers, e.g. ABB DRY-TYPE TRANSFORMER 25 kVA 480-120/240V
    # # https://electrification.us.abb.com/products/transformers-low-voltage-dry-type
    # # take siemens https://mall.industry.siemens.com/mall/de/WW/Catalog/Products/10283675
    # # size depends on the phases we want
    # # usually we have as output 400 V
    #
    # print(sum)
    # df_stacked.plot()
    # plt.show()

    transformer = Transformer(unique_id=0,
                              num_households=20)

    timestamp = start_date
    while timestamp <= end_date:
        customer = PowerCustomer(timestamp=timestamp,
                                 yearly_cons_household=4000,
                                 start_date=start_date,
                                 end_date=end_date)

        customer.set_current_load()
        customer.set_current_load_kw()

        print(customer.timestamp)
        print(transformer.transformer_capacity)
        print(customer.current_load_kw * 20)

        timestamp += interval


    # print(transformer.power_house_hold)


