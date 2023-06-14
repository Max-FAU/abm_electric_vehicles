import pandas as pd
import random



def charging_algo(capacity, df):
    distributed = 0
    while True:
        completed_agents = len(df[df['final'] == True])
        available_capacity = capacity - distributed

        remaining_agents = len(df) - completed_agents

        if remaining_agents > 0:
            charging_value_per_agent = available_capacity / remaining_agents
        else:
            charging_value_per_agent = 0

        for index, row in df.iterrows():
            if not row['final']:
                value = min(charging_value_per_agent, row['values'])
                if value >= row['values']:
                    df.loc[index, 'values'] = value
                    df.loc[index, 'final'] = True
                    distributed += value

        completed_agents_after = len(df[df['final'] == True])

        if completed_agents == completed_agents_after:
            available_capacity = capacity - distributed
            # if all agents are below in the first step it will be 4 - 4 = 0
            remaining_agents = len(df) - completed_agents_after
            if remaining_agents > 0:
                charging_value_per_agent = available_capacity / remaining_agents
                # reduce all remaining agents
                for index, row in df[df['final'] == False].iterrows():
                    df.loc[index, 'values'] = charging_value_per_agent
                    df.loc[index, 'final'] = True
                    distributed += charging_value_per_agent
            break

    return df

if __name__ == '__main__':
    capacity = 7
    n = 5

    charging_values = pd.DataFrame({"values": [random.randint(0, 10) for _ in range(n)],
                                   "final": [False] * n
                                   })
    print(charging_values)
    charging_values_new = charging_algo(capacity=capacity, df=charging_values)

    print(charging_values_new)
