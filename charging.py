from car_agent import ElectricVehicle


def calc_soc(battery_level, battery_capacity, charging_efficiency: float):
    # Calculate the energy available in the battery
    available_energy = battery_level * charging_efficiency
    # Calculate the state of charge (SoC)
    soc = available_energy / battery_capacity
    return soc


def check_current_soc(input_soc, min_soc, max_soc):
    if input_soc < min_soc:
        updated_soc = min_soc
    elif input_soc > max_soc:
        updated_soc = max_soc
    else:
        updated_soc = input_soc
    return updated_soc


if __name__ == '__main__':
    bmw_i3 = ElectricVehicle("bmw_i3")
    normal_capacity = bmw_i3.battery_capacity
    print(normal_capacity)
    current_lvl = 30
    soc_1 = calc_soc(battery_level=current_lvl,
                     battery_capacity=normal_capacity,
                     charging_efficiency=0.95)
    print(soc_1)


