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

if __name__ == '__main__':
    one_customer_base_load('2008-07-13')