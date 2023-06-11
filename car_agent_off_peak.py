from car_agent import ElectricVehicle
import pandas as pd
import datetime


class ElectricVehicleOffpeak(ElectricVehicle):
    def __init__(self, unique_id, model, car_model, start_date, end_date,
                 target_soc, max_transformer_capacity, power_customer, num_customers):
        super().__init__(unique_id, model, car_model, start_date, end_date,
                         target_soc, max_transformer_capacity, power_customer, num_customers)
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
        self.set_base_load()

        self.set_all_charging_values()
        # self.calc_charging_value()
        self.charge()
        self.set_grid_load()
        self.set_car_charging_priority()

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