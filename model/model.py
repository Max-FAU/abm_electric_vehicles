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
    def __init__(self, num_agents: int,
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
        assert num_agents <= 698, "Only 698 agents are possible"
        self.num_agents = num_agents
        self.list_models = self.generate_cars_according_to_dist()
        # self.list_models = self.generate_test_cars()

        # Return the peak load of one customer, is not added to the scheduler
        transformer_sizing = PowerCustomer(unique_id=None,
                                           model=None,
                                           yearly_cons_household=3500,
                                           start_date=self.start_date,
                                           end_date=self.end_date)

        transformer_sizing.initialize_customer()
        peak_load = transformer_sizing.get_peak_load_kw()

        # Size the transformer according to peak load and add to scheduler
        transformer = Transformer(unique_id=9999,
                                  model=self,
                                  num_households=num_agents,
                                  peak_load=peak_load)

        self.schedule.add(transformer)

        i = 0
        while i < self.num_agents:
            car_model = self.list_models[i]
            try:
                # Add Electric Vehicles to the scheduler
                car = ElectricVehicle(unique_id=i,
                                      model=self,
                                      car_model=car_model,
                                      start_date=self.start_date,
                                      end_date=self.end_date,
                                      target_soc=100,
                                      charging_algo=False)

                # Add Power Customers to the scheduler
                customer = PowerCustomer(unique_id=i + self.num_agents,
                                         model=self,
                                         yearly_cons_household=3500,
                                         start_date=start_date,
                                         end_date=end_date)

                self.schedule.add(customer)
                self.schedule.add(car)
            except Exception as e:
                print("Adding agent to model failed.")
                print(f"Error Message: {e}")

            print("...added agent number {} to the model.".format(i))
            i += 1

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
        car_models = np.random.choice(cars, size=self.num_agents, p=distribution)
        # timestamp_now = datetime.datetime.now()
        # np.savetxt('results/car_models_' + str(timestamp_now) + '.txt', car_models, fmt='%s', delimiter=' ')
        # print(len(car_names), "car names generated.")

        return car_models

    def step(self):
        self.schedule.step()
        if self.schedule.steps > 0:
            self.datacollector.collect(self)
        # print("Step ", self.schedule.steps, " completed.")


if __name__ == '__main__':
    import time
    from tqdm import tqdm
    start_time = time.time()

    start_date = '2008-07-13'
    end_date = '2008-07-20'

    time_diff = pd.to_datetime(end_date) - pd.to_datetime(start_date)
    num_intervals = int(time_diff / datetime.timedelta(minutes=15))

    model = ChargingModel(num_agents=2,
                          start_date=start_date,
                          end_date=end_date)

    for i in tqdm(range(num_intervals)):
        model.step()

    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    model_data["total_load"] = model_data["total_recharge_power"] + model_data["total_customer_load"]

    import matplotlib.pyplot as plt

    # # black and white
    # plt.style.use('grayscale')

    x_axis_time = pd.date_range(start=start_date, end=end_date, freq='15T')
    x_axis_time = x_axis_time[:-1]


    model_data['timestamp'] = x_axis_time
    model_data.set_index('timestamp', inplace=True)

    fig, ax = plt.subplots()

    model_data.plot(y=['total_recharge_power', 'total_customer_load', 'total_load', 'transformer_capacity'], ax=ax)

    plt.xlabel('Timestamp')
    plt.ylabel('kW')

    ax.set_xticks(model_data.index[::24])
    ax.set_xticklabels(model_data.index[::24].strftime('%d-%m %H:%M'), rotation=90)

    lines = ax.get_lines()

    linestyles = ['-.', '--', ':', '-']
    for i, line in enumerate(lines):
        line.set_linestyle(linestyles[i % len(linestyles)])

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False)

    legend.get_frame().set_facecolor('white')

    plt.subplots_adjust(bottom=0.3)

    plt.tight_layout()
    plt.show()

    agent_data.to_csv('results/agent_data.csv')

    print("%s seconds" % (time.time() - start_time))
