import mesa
import json
import datetime
import numpy as np
import pandas as pd
from car_agent import ElectricVehicle


def generate_cars_according_to_dist(number_of_agents):
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

    car_models = np.random.choice(cars, size=number_of_agents, p=distribution)
    # print(len(car_names), "car names generated.")

    return car_models


class ChargingModel(mesa.Model):
    def __init__(self, num_agents: int, start_date: str, end_date: str):
        """
        Simulation for charging agents
        :param num_agents: 1, 700
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
        self.list_models = generate_cars_according_to_dist(self.num_agents)

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

    def step(self):
        self.schedule.step()


if __name__ == '__main__':
    start_date = '2008-07-13'
    end_date = '2008-07-14'
    num_agents = 1

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model = ChargingModel(num_agents, start_date, end_date)

    for i in range(num_intervals):
        model.step()
