from car_agent import init_car_agent


def soc(battery_level, battery_capacity, battery_efficiency):
    # Convert battery efficiency percentage to a decimal
    battery_efficiency_decimal = battery_efficiency / 100
    # Calculate the energy available in the battery
    available_energy = battery_level * battery_efficiency_decimal
    # Calculate the state of charge (SoC)
    return (available_energy / battery_capacity) * 100


if __name__ == '__main__':
    car_expected_value = init_car_agent(name='bmw',
                                        battery_capacity='expected')
    current_lvl = 40
    soc_1 = soc(battery_level=current_lvl,
                battery_capacity=car_expected_value,
                battery_efficiency=100)
    print(soc_1)