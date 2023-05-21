class InteractionClass:
    def __init__(self, all_agents):
        # super().__init__()
        self.all_agents = all_agents

    def get_all_agents(self):
        return self.all_agents

    def get_all_priorities(self):
        all_agents = self.get_all_agents()
        all_priorities = []
        for agent in all_agents:
            all_priorities += [agent.get_charging_priority()]
        return all_priorities

    def get_all_charging_agents(self):
        car_agents = self.get_all_agents()
        car_agents_charging = []
        for agent in car_agents:
            if agent.get_charging_value() > 0:
                car_agents_charging += [agent]
        return car_agents_charging

    def get_all_charging_values(self):
        car_agents_charging = self.get_all_charging_agents()
        # get all charging values of all agents
        car_agents_charging_values = []
        for agent in car_agents_charging:
            car_agents_charging_values += [agent.charging_value]
        return car_agents_charging_values

    def get_total_charging_value(self):
        car_agents_charging_values = self.get_all_charging_values()
        # calculate the total charging values of all car agents in the model
        total_charging_value = 0
        for value in car_agents_charging_values:
            if value is not None:
                total_charging_value += value
        return total_charging_value

    def get_total_charging_power(self):
        total_charging_value = self.get_total_charging_value()
        # calculate the total charging power of all car agents in the model
        total_charging_power = total_charging_value * 4
        return total_charging_power

    def get_agents_with_charging_priority(self, priority):
        charging_agents = self.get_all_charging_agents()
        agents_with_priority = []
        for agent in charging_agents:
            if agent.get_charging_priority() == priority:
                agents_with_priority += [agent]
        return agents_with_priority

    def get_charging_power_per_agent(self, capacity, priority):
        agents_with_priority = self.get_agents_with_charging_priority(priority)
        charging_power_per_agent = capacity / len(agents_with_priority)
        return charging_power_per_agent
