import pandas as pd
import numpy as np

mobility_data = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80_test.csv"
df = pd.read_csv(mobility_data, sep=";")

# state of charge
df['e_diff'] = -1 * df['ECONSUMPTIONKWH'].diff(periods=-1)

# marking when charging is happening
df['chg'] = np.where(((df['e_diff'] > 0) & (df['ID_PANELSESSION'] == 0)), 1, 0)

chg_power = 7.2
t_delay = 2


# identify the right charger
def conditions(df):
    # if all columns > 0 then charging from Home
    if (df['chg'] == 1) and \
            (df['CHARGINGFROMHOME'] > 0) and \
            (df['CHARGINGFROMBOTH'] > 0) and \
            (df['CHARGINGFROMBOTH40'] > 0) and \
            (df['CHARGINGFROMBOTH100'] > 0):
        return 'H'
    # charging from home == 0 and all other columns > 0 then charging from Work
    elif (df['chg'] == 1) and \
            (df['CHARGINGFROMHOME'] == 0) and \
            (df['CHARGINGFROMBOTH'] > 0) and \
            (df['CHARGINGFROMBOTH40'] > 0) and \
            (df['CHARGINGFROMBOTH100'] > 0):
        return 'W'
    # charging from home and charging
    elif (df['chg'] == 1) and \
            (df['CHARGINGFROMHOME'] == 0) and \
            (df['CHARGINGFROMBOTH'] == 0) and \
            (df['CHARGINGFROMBOTH40'] > 0) and \
            (df['CHARGINGFROMBOTH100'] > 0):
        return 'XP'

    elif (df['chg'] == 1) and \
            (df['CHARGINGFROMHOME'] == 0) and \
            (df['CHARGINGFROMBOTH'] == 0) and \
            (df['CHARGINGFROMBOTH40'] == 0) and \
            (df['CHARGINGFROMBOTH100'] > 0):
        return 'XP'

    elif (df['chg'] == 1) and \
            (df['CHARGINGFROMHOME'] == 0) and \
            (df['CHARGINGFROMBOTH'] == 0) and \
            (df['CHARGINGFROMBOTH40'] == 0) and \
            (df['CHARGINGFROMBOTH100'] == 0) and \
            (df['e_diff'] > chg_power / (60 / t_delay)):
        return 'XP'

    elif (df['chg'] == 1) and \
            (df['CHARGINGFROMHOME'] == 0) and \
            (df['CHARGINGFROMBOTH'] == 0) and \
            (df['CHARGINGFROMBOTH40'] == 0) and \
            (df['CHARGINGFROMBOTH100'] == 0) and \
            (df['e_diff'] < chg_power / (60 / t_delay)):
        return 'H'
    else:
        return 'None'


df['charger'] = df.apply(lambda x: conditions(x), axis=1)

df = df[df['charger'] != 'None']
print(df)