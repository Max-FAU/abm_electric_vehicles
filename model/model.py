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
from project_paths import CAR_VALUES_PATH


class ChargingModel(Model):
    def __init__(self,
                 num_cars_normal: int,
                 num_cars_off_peak: int,
                 num_transformers: int,
                 num_customers: int,
                 start_date: str,
                 end_date: str,
                 car_charging_eff: int,
                 car_target_soc: int,
                 car_charging_algo: bool,
                 seed_value: int,
                 set_defection: bool):
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
        assert num_cars_normal <= 698, "Only 698 agents are possible"
        self.num_cars_normal = num_cars_normal
        self.num_cars_off_peak = num_cars_off_peak
        self.seed_value = seed_value
        self.list_models = self.generate_cars_according_to_dist()
        self.num_transformers = num_transformers
        self.num_customers = num_customers
        self.car_charging_eff = car_charging_eff
        self.car_target_soc = car_target_soc
        self.car_charging_algo = car_charging_algo

        # BOOL to assign defection probability
        self.set_defection = set_defection

        self.id_counter = 0

        # Return the peak load of one customer, is not added to the scheduler
        transformer_sizing = PowerCustomer(unique_id=None,
                                           model=None,
                                           yearly_cons_household=3500,
                                           start_date=self.start_date,
                                           end_date=self.end_date)

        transformer_sizing.initialize_customer()
        peak_load = transformer_sizing.get_peak_load_kw()

        i = 0
        while i < self.num_transformers:
            try:
                # Size the transformer according to peak load and add to scheduler
                unique_id_trans = self.get_next_id()
                transformer = Transformer(unique_id=unique_id_trans,
                                          model=self,
                                          num_households=num_customers,
                                          peak_load=peak_load)

                self.schedule.add(transformer)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added transformer number {} to the model.".format(i+1))
            i += 1

        o = 0
        while o < self.num_customers:
            try:
                # Add Power Customers to the scheduler
                unique_id_cust = self.get_next_id()
                customer = PowerCustomer(unique_id=unique_id_cust,
                                         model=self,
                                         yearly_cons_household=3500,
                                         start_date=start_date,
                                         end_date=end_date)

                self.schedule.add(customer)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added power customer number {} to the model.".format(o+1))
            o += 1

        j = 0
        while j < self.num_cars_normal:
            car_model = self.list_models[j]
            try:
                # Add Electric Vehicles to the scheduler
                unique_id_car = self.get_next_id()
                car = ElectricVehicle(unique_id=unique_id_car,
                                      model=self,
                                      car_model=car_model,
                                      start_date=self.start_date,
                                      end_date=self.end_date,
                                      charging_eff=self.car_charging_eff,
                                      target_soc=self.car_target_soc,
                                      charging_algo=self.car_charging_algo,
                                      seed_value=self.seed_value,
                                      defect=self.set_defection)

                self.schedule.add(car)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added normal ev number {} to the model.".format(j+1))
            j += 1

        k = 0
        while k < self.num_cars_off_peak:
            # j + k to start in the model list from the j + k entry (j has already been assigned)
            car_model = self.list_models[j + k]
            try:
                # Add Electric Vehicles to the scheduler
                unique_id_off_car = self.get_next_id()
                car = ElectricVehicleOffpeak(unique_id=unique_id_off_car,
                                             model=self,
                                             car_model=car_model,
                                             start_date=self.start_date,
                                             end_date=self.end_date,
                                             charging_eff=self.car_charging_eff,
                                             target_soc=self.car_target_soc,
                                             charging_algo=self.car_charging_algo,
                                             seed_value=self.seed_value,
                                             defect=self.set_defection)

                self.schedule.add(car)

            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added off peak ev number {} to the model.".format(k+1))
            k += 1

        self.datacollector = mesa.DataCollector(
            model_reporters={
                'total_recharge_power': self.calc_total_recharge_power,
                'total_customer_load': self.calc_total_customer_load,
                'transformer_capacity': self.calc_transformer_capacity,
                'defection_ratio': self.calc_defection_ratio
            },
            agent_reporters={
                "car_data": lambda agent: self.agent_reporter_car(
                    agent) if agent.type == 'Car' else {},
                "customer_data": lambda agent: self.agent_reporter_customer(
                    agent) if agent.type == 'Customer' else {}
            }
        )

    def get_next_id(self):
        next_id = self.id_counter
        self.id_counter += 1
        return next_id

    def agent_reporter_car(self, agent):
        return {
            "car_id": agent.car_id,
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
            "charging_power_station": agent.charging_power_station,
            "defection_probability": agent.defect,
            "wt_peer": agent.wt_peer,
            "peer_defect": agent.peer_defect,
            "past_defect_prob": agent.defect_past
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

    def calc_defection_ratio(self):
        total_defect = 0
        electric_vehicles_all = []
        for agent in self.schedule.agents:
            if isinstance(agent, ElectricVehicle):
                electric_vehicles_all.append(agent)
                if agent.defect:
                    total_defect += 1

                ratio_defectors = total_defect/len(electric_vehicles_all)
        return ratio_defectors

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

        np.random.seed(self.seed_value)
        size = (self.num_cars_normal + self.num_cars_off_peak)
        car_models = np.random.choice(cars, size=size, p=distribution)
        return car_models

# =============================================================================
#     def defection(self):
#         """function to assign defection probability"""
#         np.random.seed(self.seed_value)
#         if self.set_defection:
#             defect_prob = np.random.randint(0, 1)  # very simple 0 or 1
#             # defect_prob = 1
#         else:
#             defect_prob = 0
#         return defect_prob
#
# =============================================================================
    def step(self):
        self.schedule.step()
        if self.schedule.steps > 0:
            self.datacollector.collect(self)
        # print("Step ", self.schedule.steps, " completed.")
