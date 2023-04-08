import matplotlib.pyplot as plt
from datetime import datetime

# create plot with two y-axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# plot y1 on first y-axis
ax1.plot(data_tracking_df['timestep'], data_tracking_df['battery_power'], 'b-', label='battery capacity')
ax1.set_xlabel('Time')
ax1.set_ylabel('battery capacity in kwh')
ax1.tick_params('y', colors='b')

# plot y2 on second y-axis
ax2.plot(data_tracking_df['timestep'], data_tracking_df['consumption'], 'r-', label='consumption')
ax2.set_ylabel('consumption in kwh')
ax2.tick_params('y', colors='r')

# set xticks and rotate labels
ax1.set_xticks(data_tracking_df['timestep'][::4])
xticklabels = [datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H:%M') for x in data_tracking_df['timestep'][::4]]
ax1.set_xticklabels(xticklabels, rotation=90)

# add legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# display plot
plt.tight_layout()
plt.show()
