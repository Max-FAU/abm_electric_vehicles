from mesa import Model
import mesa
import json
import numpy as np
import pandas as pd
from car_agent import ElectricVehicle
from depreciated.transformer_agent import Transformer

import datetime


class ChargingModel(Model):
    def __init__(self, num_agents: int, start_date: str, end_date: str):
        """
        Simulation for charging agents
        :num_agents: 1 up to 698
        :start_date: min date
        :end_date: max date
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.timestamp = None

        self.schedule = mesa.time.SimultaneousActivation(self)
        assert num_agents <= 698, "Only 698 agents are possible"
        self.num_agents = num_agents
        self.list_models = self.generate_cars_according_to_dist()
        # self.list_models = self.generate_test_cars()

        transformer = Transformer(num_households=num_agents)
        transformer.initialize_transformer()
        self.max_capacity = transformer.get_max_capacity()

        i = 0
        while i < len(self.list_models):
            car_model = self.list_models[i]
            try:
                agent = ElectricVehicle(unique_id=i,
                                        model=self,
                                        car_model=car_model,
                                        start_date=self.start_date,
                                        end_date=self.end_date,
                                        target_soc=100,
                                        max_transformer_capacity=self.max_capacity)

                self.schedule.add(agent)
            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added agent number {} to the model.".format(i))
            i += 1

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "possible_capacity": lambda m: self.max_capacity,
                "total_recharge_power": self.get_total_recharge_power,
                "total_customer_load": self.get_total_customer_load
            },
            agent_reporters={
                "timestamp": lambda a: a.timestamp,
                "recharge_value": lambda a: a.charging_value,
                "battery_level": lambda a: a.battery_level,
                "soc": lambda a: a.soc
            }
        )

    def get_total_recharge_power(self):
        total_charging_power = sum([agent.charging_value * 4 for agent in self.schedule.agents])
        return total_charging_power

    def get_total_customer_load(self):
        total_base_load = sum([agent.base_load for agent in self.schedule.agents])
        return total_base_load * self.num_agents

    def generate_test_cars(self):
        file_path = 'car_models.txt'
        car_models = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                value = line.strip()
                car_models.append(value)
        return car_models

    def generate_cars_according_to_dist(self):
        with open('input/car_values.json', 'r') as f:
            data = json.load(f)

        total_cars = 0
        for name in data.keys():
            total_cars += data[name]["number"]

        cars = []
        distribution = []
        for name in data.keys():
            cars += [name]
            distribution += [data[name]["number"] / total_cars]

        car_models = np.random.choice(cars, size=self.num_agents, p=distribution)
        # timestamp_now = datetime.datetime.now()
        # np.savetxt('results/car_models_' + str(timestamp_now) + '.txt', car_models, fmt='%s', delimiter=' ')
        # print(len(car_names), "car names generated.")

        return car_models

    def step(self):
        self.schedule.step()
        if self.schedule.steps > 0:
            self.datacollector.collect(self)


if __name__ == '__main__':
    start_date = '2008-07-11'
    end_date = '2008-07-18'

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model = ChargingModel(num_agents=3,
                          start_date=start_date,
                          end_date=end_date)

    for i in range(num_intervals):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    print(model_data['total_customer_load'])
    model_data["total_load"] = model_data["total_recharge_power"] + model_data["total_customer_load"]
    # aux.set_print_options()
    # print(agent_data)
    import matplotlib.pyplot as plt
    ax = model_data.plot()
    # ax.set_ylim(0, 35)
    plt.show()
