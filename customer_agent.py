import datetime
import pandas as pd
import dask.dataframe as dd


def create_cleaned_h0_profile(self):
    df = dd.read_csv("input/h0_profile.csv")
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
    df_stacked['datetime'] = df_stacked['datetime'].apply(lambda x: x.replace(year=relevant_year))
    # set the datetime column as index
    df_stacked.set_index('datetime', inplace=True)
    # self.calc_max_peak_load(df_stacked)
    df_stacked.to_csv('cleaned_h0_profile.csv')


class PowerCustomer:
    def __init__(self,
                 yearly_cons_household,
                 start_date: str,
                 end_date: str):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.yearly_cons_household = yearly_cons_household
        self.scale = None

        self.standard_load_profile = None  # 1000 kwh yearly
        self.scaled_load_profile = None  # * 3.5

        self.current_load = None  # from base load in W
        self.current_load_kw = None  # in kW

        self.peak_load_kw = None

    def set_standard_load_profile(self):
        base_load = pd.read_csv('input/cleaned_h0_profile.csv', parse_dates=['datetime'], index_col=['datetime'])
        self.standard_load_profile = base_load

    def get_standard_load_profile(self):
        return self.standard_load_profile

    def get_yearly_cons_household(self):
        return self.yearly_cons_household

    def set_scale(self):
        consumption = self.get_yearly_cons_household()
        self.scale = consumption / 1000

    def get_scale(self):
        return self.scale

    def set_scaled_load_profile(self):
        scale = self.get_scale()
        standard_load_profile = self.get_standard_load_profile()
        self.scaled_load_profile = standard_load_profile * scale

    def get_scaled_load_profile(self):
        return self.scaled_load_profile

    def calc_peak_load(self):
        """
        This is only for comparison of the peak load of h0 profile.
        According to this study, one household has approximately 1 kW peak load.
        https://www.researchgate.net/publication/323702213_Exploring_the_impact_of_network_tariffs_on_household_electricity_expenditures_using_load_profiles_and_socio-economic_characteristics
        """
        scaled_load = self.get_scaled_load_profile()
        self.peak_load_kw = max(scaled_load['value'] / 1000)

    def initialize_customer(self):
        self.set_standard_load_profile()
        self.set_scale()
        self.set_scaled_load_profile()
        self.calc_peak_load()

    def set_current_load(self, timestamp):
        """Base load for the corresponding timestamp."""
        scaled_load_profile = self.get_scaled_load_profile()
        self.current_load = scaled_load_profile.loc[timestamp, 'value']
        self.current_load_kw = scaled_load_profile.loc[timestamp, 'value'] / 1000

    def get_current_load_w(self):
        return self.current_load

    def get_current_load_kw(self):
        return self.current_load_kw

    def get_peak_load_kw(self):
        return self.peak_load_kw


if __name__ == '__main__':
    interval = datetime.timedelta(minutes=15)
    start_date = pd.to_datetime('2008-07-13 00:00:00')
    end_date = pd.to_datetime('2008-07-13 23:00:00')

    timestamp = start_date
    customer = PowerCustomer(yearly_cons_household=3500,
                             start_date=start_date,
                             end_date=end_date)
    customer.initialize_customer()
    peak = customer.get_peak_load_kw()

    capacity = peak * 100 / 0.9 * 1.2


    # while timestamp <= end_date:
    #     customer.set_current_load(timestamp)
    #     print(timestamp)
    #     print(customer.current_load_kw)
    #     timestamp += interval
