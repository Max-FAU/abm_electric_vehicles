import mesa
from mesa import Model
from mesa.time import RandomActivation
import pandas as pd
import datetime

# Define the start and end dates for the one-week period
start_date = '2023-06-29'
end_date = '2023-07-06'

start_off_peak = pd.to_datetime('22:00:00')
end_off_peak = pd.to_datetime('06:00:00')
saturday_off_peak = pd.to_datetime('13:00:00')

# Generate the timestamps
timestamps = pd.date_range(start=start_date, end=end_date, freq='15T')
timestamps = timestamps[:-1]
start = start_off_peak.hour
end = end_off_peak.hour
saturday_start = saturday_off_peak.hour

results_df = pd.DataFrame(columns=['Timestamp', 'DayType', 'OffPeak'])

# Print the timestamps
for timestamp in timestamps:
    row = {}

    if timestamp.weekday() == 5:
        row['DayType'] = 'Saturday'

        if timestamp.hour >= saturday_start or timestamp.hour < end:
            row['OffPeak'] = True
        else:
            row['OffPeak'] = False
    elif timestamp.weekday() == 6:
        row['DayType'] = 'Sunday'
        row['OffPeak'] = True
    else:
        row['DayType'] = 'Normal Day'
        if timestamp.hour >= start or timestamp.hour < end:
            row['OffPeak'] = True
        else:
            row['OffPeak'] = False

    row['Timestamp'] = timestamp
    results_df = results_df.append(row, ignore_index=True)

results_df.to_csv('offpeak.csv')