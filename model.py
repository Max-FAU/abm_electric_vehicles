import mesa
import json
import numpy as np
import pandas as pd
from car_agent import ElectricVehicle


class ChargingModel(mesa.Model):
    def __init__(self, num_agents: int, start_date: str, end_date: str):
        """
        Simulation for charging agents
        :param num_agents: 1 upto 700
        :param start_date: min date
        :param end_date: max date
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.timestamp = None

        # TODO check if SimultaneousActivation is better
        self.schedule = mesa.time.RandomActivation(self)
        self.num_agents = num_agents
        self.list_agents = []
        self.list_models = self.generate_cars_according_to_dist()

        i = 0
        while i < len(self.list_models):
            car_model = self.list_models[i]
            try:
                agent = ElectricVehicle(unique_id=i,
                                        model=car_model,
                                        target_soc=1.0,
                                        start_date=self.start_date,
                                        end_date=self.end_date)
                self.schedule.add(agent)
            except Exception as e:
                print("Adding agent to model failed.")
            i += 1

    def generate_cars_according_to_dist(self):
        with open('car_values.json', 'r') as f:
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
        # print(len(car_names), "car names generated.")

        return car_models

    def step(self):
        self.schedule.step()