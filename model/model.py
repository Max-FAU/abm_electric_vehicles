from mesa import Model
import mesa
import json
import numpy as np
import pandas as pd
from agents.car_agent import ElectricVehicle
from agents.car_agent_off_peak import ElectricVehicleOffpeak
from agents.transformer_agent import Transformer
from agents.customer_agent import PowerCustomer
import auxiliary as aux
import datetime
from project_paths import CAR_VALUES_PATH


class ChargingModel(Model):
    def __init__(self,
                 num_cars: int,
                 num_transformers: int,
                 num_customers: int,
                 start_date: str,
                 end_date: str):
        """
        Simulation for charging agents
        :num_agents: 1 up to 698
        :start_date: min date
        :end_date: max date
        """
        super().__init__()

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.schedule = mesa.time.SimultaneousActivation(self)
        assert num_cars <= 698, "Only 698 agents are possible"
        self.num_cars = num_cars
        self.list_models = self.generate_cars_according_to_dist()
        self.num_transformers = num_transformers
        self.num_customers = num_customers

        # Return the peak load of one customer, is not added to the scheduler
        transformer_sizing = PowerCustomer(unique_id=None,
                                           model=None,
                                           yearly_cons_household=3500,
                                           start_date=self.start_date,
                                           end_date=self.end_date)

        transformer_sizing.initialize_customer()
        peak_load = transformer_sizing.get_peak_load_kw()

        j = 0
        while j < self.num_transformers:
            try:
                # Size the transformer according to peak load and add to scheduler
                unique_id_trans = j
                transformer = Transformer(unique_id=unique_id_trans,
                                          model=self,
                                          num_households=num_customers,
                                          peak_load=peak_load)

                self.schedule.add(transformer)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added transformer number {} to the model.".format(j))
            j += 1

        i = 0
        while i < self.num_cars:
            car_model = self.list_models[i]
            try:
                # Add Electric Vehicles to the scheduler
                unique_id_car = i + self.num_transformers
                car = ElectricVehicle(unique_id=unique_id_car,
                                      model=self,
                                      car_model=car_model,
                                      start_date=self.start_date,
                                      end_date=self.end_date,
                                      target_soc=100,
                                      charging_algo=False)

                self.schedule.add(car)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added electric vehicle number {} to the model.".format(i))
            i += 1

        k = 0
        while k < self.num_customers:
            try:
                # Add Power Customers to the scheduler
                unique_id_cust = k + self.num_transformers + self.num_cars
                customer = PowerCustomer(unique_id=unique_id_cust,
                                         model=self,
                                         yearly_cons_household=3500,
                                         start_date=start_date,
                                         end_date=end_date)

                self.schedule.add(customer)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added power customer number {} to the model.".format(i))
            k += 1

        self.datacollector = mesa.DataCollector(
            model_reporters={
                'total_recharge_power': self.calc_total_recharge_power,
                'total_customer_load': self.calc_total_customer_load,
                'transformer_capacity': self.calc_transformer_capacity
            },
            agent_reporters={
                "car_data": lambda agent: self.agent_reporter_car(
                    agent) if agent.type == 'Car' else {},
                "customer_data": lambda agent: self.agent_reporter_customer(
                    agent) if agent.type == 'Customer' else {}
            }
        )

    def agent_reporter_car(self, agent):
        return {
            "timestamp": agent.timestamp,
            "recharge_value": agent.charging_value,
            "battery_level": agent.battery_level,
            "soc": agent.soc,
            "charging_priority": agent.charging_priority,
            "plugged_in": agent.plugged_in,
            "battery_capacity": agent.battery_capacity,
            "trip_number": agent.trip_number,
            "deltapos": agent.deltapos,
            "cluster": agent.cluster,
            "consumption": agent.consumption,
            "panel_session": agent.panel_session,
            "terminal": agent.terminal,
            "plug_in_buffer": agent.plug_in_buffer,
            "target_soc_reached": agent.target_soc_reached,
            "charging_power_car": agent.charging_power_car,
            "charging_power_station": agent.charging_power_station
        }

    def agent_reporter_customer(self, agent):
        return {
            "timestamp": agent.timestamp,
            "consumption": agent.current_load_kw
        }

    def calc_total_recharge_power(self):
        total_recharge_value = 0
        for agent in self.schedule.agents:
            if isinstance(agent, ElectricVehicle):
                total_recharge_value += agent.charging_value
        total_recharge_power = aux.convert_kw_kwh(kwh=total_recharge_value)
        return total_recharge_power

    def calc_total_customer_load(self):
        total_customer_load = 0
        for agent in self.schedule.agents:
            if isinstance(agent, PowerCustomer):
                total_customer_load += agent.current_load_kw
        return total_customer_load

    def calc_transformer_capacity(self):
        capacity_kw = 0
        for agent in self.schedule.agents:
            if isinstance(agent, Transformer):
                capacity_kw += agent.capacity_kw
        return capacity_kw

    def generate_cars_according_to_dist(self):
        with open(CAR_VALUES_PATH, 'r') as f:
            data = json.load(f)

        total_cars = 0
        for name in data.keys():
            total_cars += data[name]["number"]

        cars = []
        distribution = []
        for name in data.keys():
            cars += [name]
            distribution += [data[name]["number"] / total_cars]

        np.random.seed(21)
        car_models = np.random.choice(cars, size=self.num_cars, p=distribution)
        # print(len(car_names), "car names generated.")

        return car_models

    def step(self):
        self.schedule.step()
        if self.schedule.steps > 0:
            self.datacollector.collect(self)
        # print("Step ", self.schedule.steps, " completed.")


if __name__ == '__main__':
    pass