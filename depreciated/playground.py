# implement transformer sizing
import datetime
import math
import random
import pandas as pd
import numpy as np
from helper import set_print_options

# load h0 house hold profile
# scale to 3.5 kwh
scale = 3.5
path = r'C:\Users\Max\PycharmProjects\mesa\h0_profile.csv'
h0_profile = pd.read_csv(path, delimiter=';')
h0_profile = h0_profile.drop(index=0)
h0_profile.index = pd.to_datetime(h0_profile['[W]']) - datetime.timedelta(minutes=15)
h0_profile = h0_profile.drop(columns=['[W]'])
h0_profile.index = h0_profile.index.strftime("%H:%M")
h0_profile = h0_profile.replace(',', '.', regex=True)
h0_profile = h0_profile.astype(float)

def calc_mean_con_year(h0_profile: pd.DataFrame):
    # check the consumption in kwh
    con_year = []
    for col in h0_profile.columns:
        result = h0_profile[col].sum() * 365 / 1000 / 4
        con_year.append(result)
    yearly_mean_con = np.mean(con_year)
    print(yearly_mean_con)

calc_mean_con_year(h0_profile)

# scale the h0_profile
h0_profile = h0_profile * scale
calc_mean_con_year(h0_profile)

set_print_options()

# Find the maximum value in each row, transformer needs to fulfill needs for max in all
max_values = h0_profile.max(axis=1)
max_values = max_values.to_frame()
# calc_mean_con_year(max_values)
# print(max_values / 1000)


# p_pt = transformer power capacity
# c_diversity = 0.2 + 0.8 / sqrt(n)   # this is only valid for private customers
# c_diverstiy = 0.5 + 0.5 / sqrt(n)
# n = expected number of customers
# f_safety = safety margin to account for the power factor and for future load growth
# c_diversity = diversity (or simultaneity) coefficient
# p_over = oversized power capacity defined by the standard size of the transformer
# https://www.sciencedirect.com/science/article/pii/S0960148117310649?via%3Dihub

num_customers = 24

list_contracted_power = []
for i in range(num_customers):
    customer_contracted_power = max_values.max() / 1000
    list_contracted_power.append(customer_contracted_power)

c_diversity = 0.2 + 0.8 / math.sqrt(num_customers)
f_safety = 1.5
p_over = 1

p_pt = c_diversity * sum(list_contracted_power) * f_safety + p_over

print(p_pt)