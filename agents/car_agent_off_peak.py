from agents.car_agent import ElectricVehicle
from agents.customer_agent import PowerCustomer
from agents.transformer_agent import Transformer
import pandas as pd
import datetime
import auxiliary as aux


class ElectricVehicleOffpeak(ElectricVehicle):
    def __init__(self, unique_id, model, car_model, start_date, end_date, target_soc, charging_algo, seed_value):
        super().__init__(unique_id, model, car_model, start_date, end_date, target_soc, charging_algo, seed_value)
        self.start_off_peak = pd.to_datetime('22:00:00')
        self.end_off_peak = pd.to_datetime('06:00:00')
        self.off_peak = False

    def set_off_peak(self):
        start = self.start_off_peak.hour
        end = self.end_off_peak.hour

        if self.timestamp.hour >= start or self.timestamp.hour < end:
            self.off_peak = True
        else:
            self.off_peak = False

    def get_off_peak(self):
        return self.off_peak

    def set_all_charging_values(self):
        plugged_in = self.get_plugged_in()
        target_soc_reached = self.get_target_soc_reached()
        off_peak = self.get_off_peak()
        if off_peak is True:
            if plugged_in is True and target_soc_reached is False:
                value = self.calc_charging_value()
                self.set_charging_value(value)
            if plugged_in is False:
                self.set_charging_value(0)
            if plugged_in is True and target_soc_reached is True:
                self.set_charging_value(0)
        else:
            self.set_charging_value(0)

    def interaction_charging_values(self):
        all_agents = self.model.schedule.agents

        transformer_capacity = 0
        # GET ALL TRANSFORMER CAPACITIES FROM ALL TRANSFORMER AGENTS
        for transformer in all_agents:
            if isinstance(transformer, Transformer):
                transformer_capacity += transformer.get_capacity_kw()

        # GET ALL LOAD FROM ALL CUSTOMER AGENTS
        customer_load = 0
        for power_customer in all_agents:
            if isinstance(power_customer, PowerCustomer):
                customer_load += power_customer.get_current_load_kw()

        # Calculate the available capacity to charge
        capacity = transformer_capacity - customer_load

        # all electric vehicle agents
        electric_vehicles = []
        for electric_vehicle in all_agents:
            if isinstance(electric_vehicle, ElectricVehicleOffpeak):
                electric_vehicles.append(electric_vehicle)

        total_charging_value = 0
        # calculate the total charging values of all car agents in the model
        for charging_value in electric_vehicles:
            if charging_value is not None:
                total_charging_value += charging_value.get_charging_value()

        # kw total charging power
        total_charging_power = aux.convert_kw_kwh(kwh=total_charging_value)

        all_priorities = []
        for prio in electric_vehicles:
            all_priorities.append(prio.get_charging_priority())

        if total_charging_power > capacity:
            highest_priority = max(all_priorities)
            lowest_priority = min(all_priorities)

            if highest_priority == lowest_priority:
                distributed = 0     # kw
                while True:
                    # get all agents that are completed / have final_charging_value = True
                    completed_charging_agents = []
                    for completed in electric_vehicles:
                        final = completed.get_final_charging_value()
                        if final:
                            completed_charging_agents.append(completed)
                    # number of not finalized charging values
                    remaining_agents = len(electric_vehicles) - len(completed_charging_agents)

                    available_capacity = capacity - distributed
                    if remaining_agents > 0:
                        charging_power_per_agent = available_capacity / remaining_agents
                    else:
                        charging_power_per_agent = 0

                    for ev in electric_vehicles:
                        if not ev.get_final_charging_value():
                            # kwh
                            charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                            # kwh
                            new_charging_value = min(charging_value_per_agent, ev.get_charging_value())
                            # kwh, kwh
                            if new_charging_value >= ev.get_charging_value():
                                ev.revert_charge()
                                ev.set_charging_value(new_charging_value)
                                ev.set_final_charging_value(True)
                                ev.charge()
                                new_charging_power = aux.convert_kw_kwh(kwh=new_charging_value)
                                distributed += new_charging_power

                    completed_charging_agents_after = []
                    for completed in electric_vehicles:
                        if completed.get_final_charging_value():
                            completed_charging_agents_after.append(completed)
                    if len(completed_charging_agents) == len(completed_charging_agents_after):
                        available_capacity = capacity - distributed
                        remaining_agents = len(electric_vehicles) - len(completed_charging_agents_after)

                        if remaining_agents > 0:
                            charging_power_per_agent = available_capacity / remaining_agents

                            for elec_vehic in electric_vehicles:
                                if not elec_vehic.get_final_charging_value():
                                    elec_vehic.revert_charge()
                                    charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                                    elec_vehic.set_charging_value(charging_value_per_agent)
                                    elec_vehic.set_final_charging_value(True)
                                    elec_vehic.charge()
                                    new_charging_power = aux.convert_kw_kwh(kwh=charging_value_per_agent)
                                    distributed += new_charging_power

                        break

                # Reset all final charging bools
                for ev in electric_vehicles:
                    ev.set_final_charging_value(False)
            else:
                distributed = 0  # kw
                # Start with the highest priority to charge proceed
                for priority in range(highest_priority, lowest_priority - 1, -1):
                    # get all agents with this priority
                    agents_with_priority = []
                    for ev in electric_vehicles:
                        if priority == ev.get_charging_priority():
                            agents_with_priority.append(ev)

                    # check if some priorities are skipped e.g. 1 car prio 5 other prio 3, 4 is then skipped
                    if len(agents_with_priority) == 0:
                        continue

                    while True:
                        # get all agents that are completed / have final_charging_value = True
                        completed_charging_agents = []
                        for completed in agents_with_priority:
                            final = completed.get_final_charging_value()
                            if final:
                                completed_charging_agents.append(completed)
                        # number of not finalized charging values
                        remaining_agents = len(agents_with_priority) - len(completed_charging_agents)

                        available_capacity = capacity - distributed
                        if remaining_agents > 0:
                            charging_power_per_agent = available_capacity / remaining_agents
                        else:
                            charging_power_per_agent = 0

                        for ev in agents_with_priority:
                            if not ev.get_final_charging_value():
                                # kwh
                                charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                                # kwh
                                new_charging_value = min(charging_value_per_agent, ev.get_charging_value())
                                # kwh, kwh
                                if new_charging_value >= ev.get_charging_value():
                                    ev.revert_charge()
                                    ev.set_charging_value(new_charging_value)
                                    ev.set_final_charging_value(True)
                                    ev.charge()
                                    new_charging_power = aux.convert_kw_kwh(kwh=new_charging_value)
                                    distributed += new_charging_power

                        completed_charging_agents_after = []
                        for completed in agents_with_priority:
                            if completed.get_final_charging_value():
                                completed_charging_agents_after.append(completed)

                        # check if there are more completed agents after loop before
                        if len(completed_charging_agents) == len(completed_charging_agents_after):
                            available_capacity = capacity - distributed
                            remaining_agents = len(agents_with_priority) - len(completed_charging_agents_after)
                            if remaining_agents > 0:
                                charging_power_per_agent = available_capacity / remaining_agents

                                for elec_vehic in agents_with_priority:
                                    if not elec_vehic.get_final_charging_value():
                                        elec_vehic.revert_charge()
                                        charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)
                                        elec_vehic.set_charging_value(charging_value_per_agent)
                                        elec_vehic.set_final_charging_value(True)
                                        elec_vehic.charge()
                                        new_charging_power = aux.convert_kw_kwh(kwh=charging_value_per_agent)
                                        distributed += new_charging_power

                            break

                # Reset all final charging bools
                for ev in electric_vehicles:
                    ev.set_final_charging_value(False)

    def step(self):
        if self.timestamp is None:
            self.timestamp = self.start_date
        else:
            # Each step add 15 minutes
            self.timestamp += datetime.timedelta(minutes=15)

        # Set mobility data for current timestamp
        self.set_data_current_timestamp()

        self.calc_new_battery_level()
        self.set_soc()
        self.set_target_soc_reached()

        self.set_off_peak()
        self.set_plug_in_status()
        self.set_charging_power_car()
        self.set_charging_power_station()

        self.set_all_charging_values()
        # self.calc_charging_value()
        self.charge()
        self.set_car_charging_priority()

        if self.charging_algo:
            # Check if the step is done for the last agent in model
            # Start the interaction
            all_agents = self.model.schedule.agents
            all_agents_ids = []
            for agent in all_agents:
                all_agents_ids += [agent.get_unique_id()]
            current_agent_id = self.get_unique_id()
            # Check if current agent id is the last id in list of ids of scheduled agents then interact
            if all_agents_ids[-1] == current_agent_id:
                # Calculate how much capacity is available for charging cars after household base load
                self.interaction_charging_values()