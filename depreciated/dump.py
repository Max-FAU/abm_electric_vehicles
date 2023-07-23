import pandas as pd

df_1 = pd.DataFrame(
    {
        "A": [1, 1, 1],
        "B": [2, 2, 2]
    }
)

df_2 = pd.DataFrame(
    {
        "C": [3, 3, 3],
        "D": [4, 4, 4]
    }
)

df_list = [df_1, df_2]

df_list_df = pd.concat(df_list, axis=1)

df_list_df['total'] = df_list_df.sum(axis=1)
df_list_df['total_2'] = df_list_df['total'] / 2

import matplotlib.pyplot as plt

plt.scatter([1] * len(df_list_df['total']), df_list_df['total'], color='blue')
plt.scatter([2] * len(df_list_df['total']), df_list_df['total'], color='blue')

custom_xticks_positions = [0, 1, 2, 3, 4, 5]
custom_xticklabels = ['', '25 EVs', '50 EVs', '150 EVs', '300 EVs', '']
plt.title('Peak Load Charging: Different Fleet Sizes')
plt.ylabel('Peak Load per EV\n[kW]')
plt.xlabel('Fleet Size')

plt.xticks(custom_xticks_positions, custom_xticklabels, rotation=90)
plt.tight_layout()
plt.show()