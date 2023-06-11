def interaction_charging_values2(self):
    available_capacity = self.get_capacity_to_charge()
    transformer_capacity = self.get_max_transformer_capacity()

    all_agents = self.model.schedule.agents
    interaction = InteractionClass(all_agents,
                                   transformer_capacity)

    interaction.set_specific_agents(ElectricVehicle)
    interaction.initialize()

    all_charging_agents = interaction.get_all_charging_agents()
    all_priorities = interaction.get_all_priorities()
    different_priorities = set(all_priorities)
    total_charging_power = interaction.get_total_charging_power()

    if total_charging_power > available_capacity:
        # Check if all agents have the same priority
        if len(different_priorities) > 1:
            print("different prios")
            highest_priority = max(all_priorities)
            lowest_priority = min(all_priorities)

            current_charging_power = 0
            for priority in range(highest_priority, lowest_priority - 1, -1):
                interaction.set_priority(priority)
                interaction.set_agents_with_charging_priority()
                interaction.set_charging_power_per_agent()

                interaction.adjust_charging_values()

        else:
            print("all same prio")
            interaction.set_priority(different_priorities)
            interaction.set_agents_with_charging_priority()
            interaction.set_charging_power_per_agent()

            interaction.adjust_charging_values()

    else:
        print("next timestamp")
