from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import random

class ChargingStation(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.status = "unoccupied"
        self.charge_start_time = 0
        self.charge_duration = 0

    def charge(self):
        self.status = "occupied"
        self.charge_start_time = self.model.schedule.time
        self.charge_duration = random.randint(1, 3) * 60  # charging duration in minutes

    def step(self):
        if self.status == "occupied":
            if self.model.schedule.time >= (self.charge_start_time + self.charge_duration):
                self.status = "unoccupied"

class EV(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.charge_needed = random.randint(20, 80)  # charge needed in percentage
        self.charge_received = 0

    def charge(self):
        if self.charge_received < self.charge_needed:
            for station in self.model.schedule.agents:
                if station.status == "unoccupied":
                    station.charge()
                    break

    def step(self):
        self.charge()
        self.charge_received = sum([station.charge_duration for station in self.model.schedule.agents if station.status == "occupied"]) / 60

class EVChargingModel(Model):
    def __init__(self, num_stations, num_cars):
        self.num_stations = num_stations
        self.num_cars = num_cars
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(1, 1, False)

        # create charging stations
        for i in range(self.num_stations):
            station = ChargingStation(i, self)
            self.schedule.add(station)
            x = self.random.randint(0, self.grid.width-1)
            y = self.random.randint(0, self.grid.height-1)
            self.grid.place_agent(station, (x, y))

        # create electric vehicles
        for i in range(self.num_cars):
            ev = EV(i, self)
            self.schedule.add(ev)

    def step(self):
        self.schedule.step()


if __name__ == '__main__':
    pass