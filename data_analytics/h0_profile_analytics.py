import pandas as pd
import matplotlib.pyplot as plt
h0 = r'C:\Users\Max\PycharmProjects\mesa\input\cleaned_h0_profile.csv'

h0 = pd.read_csv(h0, parse_dates=['datetime'], index_col=['datetime'])
h0 = h0/1000
h0 = h0.sort_values(by='datetime')


h0_scaled = h0 * 3.5
h0_scaled = h0_scaled.sort_values(by='datetime')
start_date = '2008-07-13'
end_date = '2008-07-27'

h0_scaled_filtered = h0_scaled.loc[(h0_scaled.index >= start_date) & (h0_scaled.index < end_date)]
h0_scaled_filtered = h0_scaled_filtered.sort_values(by='datetime')

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# First subplot
axes[0].plot(h0.index, h0, color='darkgrey')
axes[0].set_ylim(0, 1)
axes[0].set_title('Step 1 \n Original H0 Profile')
axes[0].set_xlabel('Datetime')
axes[0].set_ylabel('Load [kW]')
axes[0].tick_params(axis='x', rotation=90)

# Second subplot
axes[1].plot(h0.index, h0_scaled, color='darkgrey')
axes[1].set_ylim(0, 1)
axes[1].set_title('Step 2 \n Scaled H0 Profile')
axes[1].set_xlabel('Datetime')
axes[1].set_ylabel('Load [kW]')
axes[1].tick_params(axis='x', rotation=90)

# Third subplot
axes[2].plot(h0_scaled_filtered.index, h0_scaled_filtered, color='darkgrey')
axes[2].set_ylim(0, 1)
axes[2].set_title('Step 3 \n Filtered and Scaled H0 Profile')
axes[2].set_xlabel('Datetime')
axes[2].set_ylabel('Load [kW]')
axes[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig("H0_profile_analytics", dpi=300)
plt.show()