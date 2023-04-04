# Import necessary libraries
import random
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid

# Define a charging station agent
class ChargingStation(Agent):
    def __init__(self, unique_id, model, max_power):
        super().__init__(unique_id, model)
        self.max_power = max_power
        self.available_power = max_power
        self.queue = []

    def charge(self, ev_agent):
        # Check if there is enough power available
        if ev_agent.required_power <= self.available_power:
            self.available_power -= ev_agent.required_power
            ev_agent.start_charging(self.model.schedule.time)
        else:
            self.queue.append(ev_agent)

    def unplug(self, ev_agent):
        self.available_power += ev_agent.required_power
        if self.queue:
            self.charge(self.queue.pop(0))

    def step(self):
        # Check if there are any EVs waiting to charge
        if self.queue:
            self.charge(self.queue.pop(0))

# Define an EV agent
class EV(Agent):
    def __init__(self, unique_id, model, charging_power, charging_start_time, charging_end_time):
        super().__init__(unique_id, model)
        self.charging_power = charging_power
        self.charging_start_time = charging_start_time
        self.charging_end_time = charging_end_time
        self.required_power = (charging_end_time - charging_start_time) * charging_power
        self.charge_start_time = None

    def start_charging(self, current_time):
        self.charge_start_time = current_time

    def stop_charging(self):
        self.charge_start_time = None

    def step(self):
        # Check if charging time has ended
        if self.charge_start_time is not None and self.model.schedule.time >= self.charging_end_time:
            self.stop_charging()
            self.model.station.unplug(self)

# Define the charging model
class ChargingModel(Model):
    def __init__(self, num_ev, max_power, ev_charging_power, sim_duration):
        self.num_ev = num_ev
        self.max_power = max_power
        self.ev_charging_power = ev_charging_power
        self.sim_duration = sim_duration
        self.schedule = RandomActivation(self)
        self.station = ChargingStation(1, self, max_power)
        self.grid = MultiGrid(1, 1, False)

        # Create EVs with random charging times and add them to the schedule
        for i in range(num_ev):
            charging_start_time = random.uniform(0, sim_duration)
            charging_end_time = random.uniform(charging_start_time, sim_duration)
            ev = EV(i, self, ev_charging_power, charging_start_time, charging_end_time)
            self.schedule.add(ev)

    def step(self):
        # Advance the model by one step
        self.schedule.step()

    def run_model(self):
        # Run the model until the end of the simulation
        for i in range(int(self.sim_duration)):
            self.step()

# Create and run the model
model = ChargingModel(num_ev=10, max_power=100, ev_charging_power=5, sim_duration=24)
model.run_model()

# Print the total power used by the EVs and the amount of power remaining at the station
ev_power_usage = sum([ev.required_power for ev in model.schedule.agents if isinstance(ev, EV)])
station_power_remaining = model.station.available_power
print(f'Total power used by the EVs: {ev_power_usage}')
print(f'Power remaining at the station: {station_power_remaining}')
