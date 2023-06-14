import pandas as pd
import random


capacity = 5000
n = 1000

df_1 = pd.DataFrame({"values": [random.randint(0, 7) for _ in range(n)],
                     "final": [False] * n
                     })
distributed = 0

while True:
    completed_agents = len(df_1[df_1['final'] == True])
    available_capacity = capacity - distributed

    remaining_agents = len(df_1) - completed_agents

    if remaining_agents > 0:
        charging_value_per_agent = available_capacity / remaining_agents
    else:
        charging_value_per_agent = 0

    for index, row in df_1.iterrows():
        if not row['final']:
            value = min(charging_value_per_agent, row['values'])
            if value >= row['values']:
                df_1.loc[index, 'values'] = value
                df_1.loc[index, 'final'] = True
                distributed += value

    completed_agents_after = len(df_1[df_1['final'] == True])

    if completed_agents == completed_agents_after:
        available_capacity = capacity - distributed
        # if all agents are below in the first step it will be 4 - 4 = 0
        remaining_agents = len(df_1) - completed_agents_after
        if remaining_agents > 0:
            charging_value_per_agent = available_capacity / remaining_agents
            # reduce all remaining agents
            for index, row in df_1[df_1['final'] == False].iterrows():
                df_1.loc[index, 'values'] = charging_value_per_agent
                df_1.loc[index, 'final'] = True
                distributed += charging_value_per_agent
        break

print(df_1)
print(capacity)
print(distributed)
