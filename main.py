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
        for car_model in self.list_models:
            agent = ElectricVehicle(model=car_model,
                                    unique_id=i,
                                    target_soc=1.0,
                                    start_date=self.start_date,
                                    end_date=self.end_date)

            self.schedule.add(agent)
            i += 1
            agent.set_timestamp(self.timestamp)

    def step(self):

        if self.timestamp is None:
            self.timestamp = self.start_date
        else:
            # each step add 15 minutes
            self.timestamp = self.timestamp + datetime.timedelta(minutes=15)

        self.schedule.step()


if __name__ == '__main__':
    model = ChargingModel(30, '2008-07-13', '2008-07-14')
    # print(model.list_agents)

    # for timestamp in mobility_data.index:
    #     # print(timestamp)
    #     # bmw_i3.next_trip_needs(mobility_data, timestamp)
    #     model.step()