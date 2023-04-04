import pandas as pd
import matplotlib.pyplot as plt


# filter only when possible to charge ID_PANELSESSION == 0 means charging is possible
# keeps data only for one trip testing purpose
def create_trip_df(df, trip_no=None):
    if trip_no is None:
        mobility_data = df
    else:
        mobility_data = df[df['TRIPNUMBER'] == trip_no]
    return mobility_data

# keep only relevant columns in the dataframe
def keep_relevant_columns(df):
    df = df[['TIMESTAMP', 'TRIPNUMBER', 'ID_PANELSESSION', 'LONGITUDE', 'LATITUDE']]
    # df = df[df['ID_PANELSESSION'] == 0]
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    return df

# calculate the time a car could possibly charge per trip
def charging_time_per_trip(df):
    df['diff'] = df['TIMESTAMP'].diff()
    # print(df)
    # filter out everything when car is moving and charging not possible
    df_new = df[df['ID_PANELSESSION'] == 0]
    charging_time_per_trip = df_new.groupby(['TRIPNUMBER'])['diff'].sum()
    print(charging_time_per_trip)

if __name__ == '__main__':
    mobility_data = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    mobility_data = pd.read_csv(mobility_data)

    mobility_data = create_trip_df(mobility_data, trip_no=1)
    mobility_data = keep_relevant_columns(mobility_data)
    # print(mobility_data)
    # mobility_data.to_csv('mobility_data_trip1_test.csv')

    # print charging duration per trip
    charging_time_per_trip(mobility_data)

    # create mobility data with 15 min timesteps
    # what happens when mobility data is taken

    # print(type(mobility_data['TIMESTAMP'].iloc[1]))
