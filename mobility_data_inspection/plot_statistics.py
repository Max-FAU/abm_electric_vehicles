import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mobility_data_statistics.csv')
print(df.columns)
df['timestamp_density'] = df['timestamp_density'] / 60

min1 = min(df['average_trip_distance'])
max1 = max(df['average_trip_distance'])

min2 = min(df['median_trip_distance'])
max2 = max(df['median_trip_distance'])

min = min(min1, min2, 0)
max = int(round(max(max1, max2), -4))

# Specify the bin edges
bin_edges = list(range(min, max, 1000))

list = [df['median_trip_distance'], df['average_trip_distance']]

for entry in list:
    # Plot the histogram
    plt.hist(entry, bins=bin_edges)
    plt.ylim(0, 200)
    plt.show()

labels = df['label'].unique()
fig, ax = plt.subplots()
anzahl_priv = len(df[df['label'] == 'private'])
anzahl_commercial = len(df[df['label'] == 'commercial'])
counts = [anzahl_priv, anzahl_commercial]
plt.bar(labels, counts)
plt.show()