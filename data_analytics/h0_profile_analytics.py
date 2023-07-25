import pandas as pd
import matplotlib.pyplot as plt


def plot_steps():
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

def plot_h0():
    h0 = r'C:\Users\Max\PycharmProjects\mesa\input\cleaned_h0_profile.csv'

    h0 = pd.read_csv(h0, parse_dates=['datetime'], index_col=['datetime'])
    h0 = h0/1000
    h0 = h0.sort_values(by='datetime')
    fig, ax = plt.subplots(figsize=(12, 6))
    h0_scaled = h0 * 3.5
    h0_scaled = h0_scaled.sort_values(by='datetime')
    start_date = '2008-07-13'
    end_date = '2008-07-27'

    h0_scaled_filtered = h0_scaled.loc[(h0_scaled.index >= start_date) & (h0_scaled.index < end_date)]
    h0_scaled_filtered = h0_scaled_filtered.sort_values(by='datetime')
    print(h0_scaled_filtered)
    h0_scaled_filtered['hour'] = h0_scaled_filtered.index.hour
    h0_scaled_filtered['minute'] = h0_scaled_filtered.index.minute
    h0_scaled_filtered['weekday'] = h0_scaled_filtered.index.weekday
    h0_scaled_filtered['day_type'] = h0_scaled_filtered['weekday'].apply(
        lambda x: 'Weekday' if x < 5 else 'Weekend')
    df_grouped = h0_scaled_filtered[h0_scaled_filtered['day_type'] == 'Weekday']

    df_grouped = df_grouped.groupby(['hour', 'minute']).mean()
    df_grouped.reset_index(inplace=True)
    # df_grouped = df_grouped[['hour', 'minute', 'average', 'percentile_5', 'percentile_95']]
    df_grouped['minute'] = df_grouped['minute'].astype(str).str.zfill(2)
    df_grouped['time'] = df_grouped['hour'].astype(str) + ':' + df_grouped['minute'].astype(str)
    print(df_grouped)
    df_grouped.plot(x='time', y='value', ax=ax, label='Average Customer Load', color='blue', linestyle='dashed',
                    linewidth=1.5)

    ax.set_title('H0 Profile Weekday')
    ax.set_ylabel('Customer Load\n[kW]')
    ax.legend().remove()
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 0.85)

    tick_positions = df_grouped.index[::4]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(df_grouped['time'][::4], rotation=90)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # plot_steps()
    plot_h0()