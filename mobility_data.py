import pandas as pd


class MobilityDataAggregator:
    def __init__(self,
                 raw_mobility_data: pd.DataFrame,
                 start_date: str,
                 end_date: str):

        self.raw_mobility_data = raw_mobility_data
        self.df_limited_time = pd.DataFrame()
        self.df_processed = pd.DataFrame()

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.median_trip_length = None
        self.prepare_mobility_data()
        self.set_median_trip_len()

    def _create_df_limited_time(self):
        raw_mobility_data = self.get_raw_df()
        # Convert the TIMESTAMP column to timestamp to compare it later
        raw_mobility_data['TIMESTAMP'] = pd.to_datetime(raw_mobility_data['TIMESTAMP'])

        raw_start_date = pd.to_datetime(min(self.raw_mobility_data['TIMESTAMP']))
        raw_end_date = pd.to_datetime(max(self.raw_mobility_data['TIMESTAMP']))

        if self.start_date < raw_start_date:
            print('Start date: {} is too early for this data set.'.format(raw_start_date))
        if self.end_date > raw_end_date:
            print('End date: {} is too late for this data set.'.format(raw_end_date))

        # to avoid errors when setting data to timestamp, each timestamp has to present in the mobility file
        # create a dataframe only with all timestamps between start and end date
        timestamps = pd.date_range(start=self.start_date,
                                   end=self.end_date,
                                   freq='15T')

        df_timestamps = pd.DataFrame({'TIMESTAMP': timestamps})

        # left join the dataframe to only assign values to existing timestamps
        self.df_limited_time = pd.merge(df_timestamps, raw_mobility_data, on='TIMESTAMP', how='left')

        # filter the raw mobility data according to start and end time
        # self.df_limited_time = raw_mobility_data.loc[(raw_mobility_data['TIMESTAMP'] >= self.start_date) &
        #                                              (raw_mobility_data['TIMESTAMP'] < self.end_date)]

    def _aggregate_15_min_steps(self):
        # calculate the total energy demand in that 15 minutes
        self.df_limited_time = pd.DataFrame(self.df_limited_time)
        self.df_limited_time.set_index('TIMESTAMP', inplace=True)

        # Refactor, columns are already in json relevant columns
        self.df_processed = self.df_limited_time.resample('15T', closed='left').agg(
            {'ECONSUMPTIONKWH': 'sum',  # charging dependent on this
             'TRIPNUMBER': 'min',
             'ID_PANELSESSION': 'max',  # charging dependent on this
             'ID_TERMINAL': 'first',  # car_id
             'CLUSTER': 'min',  # home / work location
             'DELTAPOS': 'sum'}
        )
        # return self.df_processed
        self.df_processed = self.df_processed.ffill()  # fill missing values with the following value
        self.df_processed = self.df_processed.bfill()  # since ffill does not work if the first row is missing

    def _data_cleaning(self):
        try:
            # Every row needs an entry in tripno, econ
            self.df_processed = self.df_processed.dropna(subset=['TRIPNUMBER', 'ECONSUMPTIONKWH'])
        except ValueError:
            print("Dropping NAN values in dataframe failed.")

        # All timestamps need to match the given format
        timestamp_format = '%Y-%m-%d %H:%M'
        try:
            pd.to_datetime(self.df_processed.index, format=timestamp_format, errors='coerce').notnull().all()
        except ValueError:
            print("Timestamp index error - wrong format.")

    def prepare_mobility_data(self):
        self._create_df_limited_time()
        self._aggregate_15_min_steps()
        self._data_cleaning()

    def set_median_trip_len(self):
        trip_df = self.df_processed.groupby('TRIPNUMBER').sum()
        self.median_trip_length = trip_df['DELTAPOS'].median()

    def get_raw_df(self):
        return self.raw_mobility_data

    def get_limited_time_df(self):
        return self.df_limited_time

    def get_processed_df(self):
        return self.df_processed
