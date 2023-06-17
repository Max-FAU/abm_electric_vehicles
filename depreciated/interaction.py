import auxiliary as aux


class InteractionClass:
    def __init__(self,
                 all_agents):

        self.all_agents = all_agents
        # self.transformer_capacity = transformer_capacity

        self.specific_agents = []
        self.all_priorities = []

        self.car_agents_charging = []
        self.car_agents_charging_values = []
        self.total_charging_value = 0
        self.total_charging_power = None

        self.priority = None
        self.agents_with_priority = []
        self.charging_power_per_agent = None

        self.processed_agents = []
        self.additional_charging_power = 0
        self.left_over = []

    def get_all_agents(self):
        return self.all_agents

    def get_available_capacity(self):
        return self.available_capacity

    def get_specific_agents(self):
        return self.specific_agents

    def set_specific_agents(self, agent_class):
        all_agents = self.get_all_agents()
        # Filter to keep only ElectricVehicles (Filter out transformers)
        for agent in all_agents:
            if isinstance(agent, agent_class):
                self.specific_agents.append(agent)

    def set_all_priorities(self):
        all_agents = self.get_specific_agents()
        for agent in all_agents:
            self.all_priorities.append(agent.get_charging_priority())

    def get_all_priorities(self):
        return self.all_priorities

    def set_all_charging_agents(self):
        car_agents = self.get_specific_agents()
        for agent in car_agents:
            if agent.get_charging_value() > 0:
                self.car_agents_charging.append(agent)

    def get_all_charging_agents(self):
        return self.car_agents_charging

    def set_all_charging_values(self):
        car_agents_charging = self.get_all_charging_agents()
        # get all charging values of all agents
        for agent in car_agents_charging:
            if agent.get_charging_value() > 0:
                self.car_agents_charging_values.append(agent.get_charging_value())

    def get_all_charging_values(self):
        return self.car_agents_charging_values

    def set_total_charging_value(self):
        car_agents_charging_values = self.get_all_charging_values()
        # calculate the total charging values of all car agents in the model
        for value in car_agents_charging_values:
            if value is not None:
                self.total_charging_value += value

    def get_total_charging_value(self):
        return self.total_charging_value

    def set_total_charging_power(self):
        total_charging_value = self.get_total_charging_value()
        # calculate the total charging power of all car agents in the model
        self.total_charging_power = aux.convert_kw_kwh(kwh=total_charging_value)

    def get_total_charging_power(self):
        return self.total_charging_power

    def set_priority(self, priority):
        self.priority = priority

    def get_priority(self):
        return self.priority

    def set_agents_with_charging_priority(self):
        charging_agents = self.get_all_charging_agents()
        priority = self.get_priority()
        for agent in charging_agents:
            if agent.get_charging_priority() == priority:
                self.agents_with_priority.append(agent)

    def get_agents_with_charging_priority(self):
        return self.agents_with_priority

    def set_charging_power_per_agent(self):
        capacity = self.get_available_capacity()
        agents_with_priority = self.get_agents_with_charging_priority()
        self.charging_power_per_agent = capacity / len(agents_with_priority)

    def get_charging_power_per_agent(self):
        return self.charging_power_per_agent

    def add_processed_agents(self, agent):
        self.processed_agents.append(agent)

    def compare_charging_power(self):
        agents_with_priority = self.get_agents_with_charging_priority()
        charging_power_per_agent = self.get_charging_power_per_agent()

        for agent in agents_with_priority:
            charging_value = agent.get_charging_value()
            charging_power = aux.convert_kw_kwh(kwh=charging_value)

            if charging_power_per_agent >= charging_power:
                self.additional_charging_power += charging_power_per_agent - charging_power
                self.available_capacity -= charging_power
                self.add_processed_agents(agent)

    def distribute_rest_capacity(self):
        processed = self.processed_agents
        agents_with_priority = self.get_agents_with_charging_priority()

        self.left_over = [x for x in agents_with_priority if x not in processed]

        self.available_capacity = self.available_capacity + self.additional_charging_power
        # Calc new charging values
        self.charging_power_per_agent = self.available_capacity / len(self.left_over)

    def compare_again(self):
        processed = self.processed_agents
        agents_with_priority = self.get_agents_with_charging_priority()
        charging_power_per_agent = self.get_charging_power_per_agent()

        self.left_over = [x for x in agents_with_priority if x not in processed]

        for agent in self.left_over:
            charging_value = agent.get_charging_value()
            charging_power = aux.convert_kw_kwh(kwh=charging_value)

            if charging_power_per_agent >= charging_power:
                self.additional_charging_power += charging_power_per_agent - charging_power
                self.available_capacity -= charging_power
                self.add_processed_agents(agent)

    def adjust_charging_values(self):
        while len(self.left_over) > 0:
            self.compare_charging_power()
            self.distribute_rest_capacity()
            self.compare_again()

    def initialize(self):
        self.set_all_priorities()
        self.set_all_charging_agents()
        self.set_all_charging_values()
        self.set_total_charging_value()
        self.set_total_charging_power()
