import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

#
# # create plot with two y-axes
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# # plot y1 on first y-axis
# ax1.plot(data_tracking_df['timestep'], data_tracking_df['battery_power'], 'b-', label='battery capacity')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('battery capacity in kwh')
# ax1.tick_params('y', colors='b')
#
# # plot y2 on second y-axis
# ax2.plot(data_tracking_df['timestep'], data_tracking_df['consumption'], 'r-', label='consumption')
# ax2.set_ylabel('consumption in kwh')
# ax2.tick_params('y', colors='r')
#
# # set xticks and rotate labels
# ax1.set_xticks(data_tracking_df['timestep'][::4])
# xticklabels = [datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').strftime('%H:%M') for x in data_tracking_df['timestep'][::4]]
# ax1.set_xticklabels(xticklabels, rotation=90)
#
# # add legend
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
#
# # display plot
# plt.tight_layout()
# plt.show()
#
# #### next plot
#
# path1 = r"I:\Max_Mobility_Profiles\quarterly_simulation\quarterly_simulation_80.csv"
# path2 = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
# try:
#     mobility_data = pd.read_csv(path1)
# except FileNotFoundError:
#     mobility_data = pd.read_csv(path2)
#
# mobility_data = prepare_mobility_data(df=mobility_data,
#                                       starting_date='2008-07-12 00:00:00',
#                                       days=1)
#
# mobility_data_aggregated = aggregate_15_min_steps(mobility_data)
#
# plot = False
# if plot:
#     try:
#         test = pd.read_csv(path1)
#     except FileNotFoundError:
#         test = pd.read_csv(path2)
#
#     test = prepare_mobility_data(df=test,
#                                  starting_date='2008-07-12 00:00:00',
#                                  days=14)
#
#     test.set_index('TIMESTAMP', inplace=True)
#
#     charging_1 = (test['ECONSUMPTIONKWH'] <= 0)
#     charging_2 = (mobility_data_aggregated['ECONSUMPTIONKWH'] <= 0)
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
#
#     ax1.plot(test['ECONSUMPTIONKWH'], color='blue')
#     ax1.set_ylabel('baseline_value')
#
#     ax1.fill_between(test.index,
#                      np.min(mobility_data_aggregated['ECONSUMPTIONKWH']),
#                      np.max(mobility_data_aggregated['ECONSUMPTIONKWH']) * 1.1,
#                      where=charging_1, alpha=0.3, color='green')
#
#     ax2.plot(mobility_data_aggregated['ECONSUMPTIONKWH'], color='black')
#     ax2.set_ylabel('aggregated_value')
#
#     ax2.fill_between(mobility_data_aggregated.index,
#                      np.min(mobility_data_aggregated['ECONSUMPTIONKWH']),
#                      np.max(mobility_data_aggregated['ECONSUMPTIONKWH']) * 1.1,
#                      where=charging_2, alpha=0.3, color='green')
#
#     ylim_max_value = max(max(test['ECONSUMPTIONKWH']),
#                          max(mobility_data_aggregated['ECONSUMPTIONKWH'])) * 1.1
#
#     for ax in [ax1, ax2]:
#         ax.tick_params(axis='x', labelrotation=90)
#         ax.set_ylim(0, ylim_max_value)
#
#     plt.tight_layout()
#     plt.show()
#
# print(mobility_data_aggregated)


#############################

def create_plot(start_date, end_date, df_results: pd.DataFrame):
    # # black and white
    # plt.style.use('grayscale')
    x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
    x_axis_time = x_axis_time[:-1]

    df_results['timestamp'] = x_axis_time
    df_results.set_index('timestamp', inplace=True)

    fig, ax = plt.subplots()

    df_results.plot(y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'], ax=ax)

    plt.xlabel('Timestamp')
    plt.ylabel('kW')

    ax.set_xticks(df_results.index[::24])
    ax.set_xticklabels(df_results.index[::24].strftime('%d-%m %H:%M'), rotation=90)

    lines = ax.get_lines()

    linestyles = ['-.', '--', ':', '-']
    for i, line in enumerate(lines):
        line.set_linestyle(linestyles[i % len(linestyles)])

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)

    legend.get_frame().set_facecolor('white')

    plt.subplots_adjust(bottom=0.3)

    plt.tight_layout()
    plt.show()
