import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mobility_data_statistics.csv')
print(df.columns)
df['timestamp_density'] = df['timestamp_density'] / 60

min1 = min(df['average_trip_distance'])
max1 = max(df['average_trip_distance'])

min2 = min(df['median_trip_distance'])
max2 = max(df['median_trip_distance'])

min = min(min1, min2, 0)
max = int(round(max(max1, max2), -4))

plot = False
if plot:
    # list = [df['median_trip_distance'], df['average_trip_distance']]
    sns.violinplot(data=df[['median_trip_distance', 'average_trip_distance']])

    # # Plotting the violin plots in the second subplot
    # sns.violinplot(data=df['average_trip_distance'], ax=axs[1], color='green')

    plt.title('Violin Chart Comparison')
    plt.legend()
    plt.show()


    # labels = df['label'].unique()
    # fig, ax = plt.subplots()
    # anzahl_priv = len(df[df['label'] == 'private'])
    # print(anzahl_priv)
    # anzahl_commercial = len(df[df['label'] == 'commercial'])
    # counts = [anzahl_priv, anzahl_commercial]
    # plt.bar(labels, counts)
    # plt.show()

def statistics(data):
    data = data.copy()
    # Calculate the average median trip length
    avg_median_trip_length = data['median_trip_distance'].mean()
    # Calculate the average average trip length
    avg_avg_trip_length = data['average_trip_distance'].mean()
    # Calculate the average density
    avg_density = data['timestamp_density'].mean()
    data['earliest_entry'] = data['earliest_entry'].str.slice(0, 10)
    data['latest_entry'] = data['latest_entry'].str.slice(0, 10)
    data['entry_tuple'] = list(zip(data['earliest_entry'], data['latest_entry']))
    count_tuples = data['entry_tuple'].value_counts()
    most_common_tuple = count_tuples.idxmax()
    avg_entries = data['length'].mean()
    print(len(data), most_common_tuple, round(avg_entries,2), round(avg_median_trip_length,2), round(avg_avg_trip_length,2), round(avg_density,2))

print('Entries, Most common start and end, avg entries, avg median trip length, avg trip length, avg timestamps density in mins')
statistics(df)
statistics(df[df['label'] == 'commercial'])
statistics(df[df['label'] == 'private'])
# df = df.groupby