import mesa
from mesa import Model
from mesa.time import RandomActivation
import pandas as pd
import datetime


def interaction_charging_values(self):
    max_capacity = self.get_capacity_to_charge()

    # This action is done in every step
    all_agents = self.model.schedule.agents

    # Filter to keep only ElectricVehicles (Filter out transformers)
    car_agents = []
    for agent in all_agents:
        if isinstance(agent, ElectricVehicle):
            car_agents.append(agent)

    # Enter the interaction class to reduce interaction charging values method
    model_interaction = InteractionClass(car_agents)
    all_charging_agents = model_interaction.get_all_charging_agents()
    all_priorities = model_interaction.get_all_priorities()
    total_charging_power = model_interaction.get_total_charging_power()

    if total_charging_power > max_capacity:     # check if total_charging_power is higher than available capacity
        highest_priority = max(all_priorities)    # get highest priority
        lowest_priority = min(all_priorities)       # get lowest priority
        agents_higher_priority = []

        available_capacity = max_capacity       # maximum capacity is available capacity

        # Starting with the highest priority to charge
        for priority in range(highest_priority, lowest_priority - 1, -1):
            agents_priority = model_interaction.get_agents_with_charging_priority(priority)  # get all agents having the priority

            sub_total_charging_power = 0    # sum up the charging power for these agents

            for agent in agents_priority:   # for each agent having the priority
                charging_value = agent.get_charging_value()     # get the charging value
                charging_power = aux.convert_kw_kwh(kwh=charging_value)  # get the charging power
                sub_total_charging_power += charging_power  # add the charging power to the sum of this priority

                # this is checked agent after agent
                if sub_total_charging_power > available_capacity: # check if the sum of the current priority is already higher than max_capacity
                    # if it is higher, reduce everything in this priority by the same lvl
                    # for that calculate the available capacity per agents with this priority
                    charging_power_per_agent = model_interaction.get_charging_power_per_agent(available_capacity,
                                                                                              priority)
                    charging_value_per_agent = aux.convert_kw_kwh(kw=charging_power_per_agent)  # convert to kwh

                    charging_value_to_distribute = 0
                    agents_exceeding_charging_value = []
                    num_agents_priority = len(agents_priority)

                    for agent in agents_priority:     # Loop alle agenten mit der Priorität
                        # check all charging values and check if the charging value is higher than what should be available for him
                        if agent.get_charging_value() < charging_value_per_agent:   # TODO CHECK THIS
                            # if the charging_value is smaller than the charging_value per agent
                            # this is done because not every car has the same charging value??????
                            charging_value_to_distribute += (charging_value_per_agent - agent.get_charging_value())
                            # find the agents where the agents are exceeding the charging
                            agents_exceeding_charging_value.append(agent)

                    # Dann müssen andere Agenten erhöht werden
                    other_agents_increase = charging_value_to_distribute / num_agents_priority

                    for agent in agents_priority:
                        agent.revert_charge()
                        if agent in agents_exceeding_charging_value:
                            charging_value = min(agent.get_charging_value(), charging_value_per_agent)
                        else:
                            charging_value = charging_value_per_agent + other_agents_increase
                        if charging_value > agent.get_charging_value():
                            charging_value = min(agent.get_charging_value(), charging_value)

                            # Schauen ob der neue charging value mit der "umlage" größer ist als der
                            # alte, wenn größer als der alte, dann den alten charging value nehmen
                            # differenz wieder berechnen und auf die noch nicht processed agents addieren
                            # wenn der letzte agent dass dann nicht mehr aufnehmen kann, wegfallen lassen.

                        agent.set_charging_value(value=charging_value)
                        agent.charge()

                    # check if higher priorities were already processed
                    for agent in all_charging_agents:
                        if agent not in agents_priority and agent not in agents_higher_priority:
                            agent.revert_charge()
                            agent.set_charging_value(value=0)
                            agent.charge()
                    break

                agents_higher_priority.append(agent)
            available_capacity -= sub_total_charging_power

def get_right_charging_power_station():
    # TODO Maybe implement these charging values, but with what logic?
    chose = [3.7, 7.2, 11, 22]


if __name__ == '__main__':
    path = r'C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_test.csv'
    df = pd.read_csv(path, delimiter=";")
    print(df)