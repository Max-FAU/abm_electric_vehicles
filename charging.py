from car_agent import ElectricVehicle

def soc(battery_level, battery_capacity, battery_efficiency):
    # Convert battery efficiency percentage to a decimal
    battery_efficiency_decimal = battery_efficiency / 100
    # Calculate the energy available in the battery
    available_energy = battery_level * battery_efficiency_decimal
    # Calculate the state of charge (SoC)
    return (available_energy / battery_capacity) * 100


if __name__ == '__main__':

    bmw_i3 = ElectricVehicle("bmw_i3")
    normal_capacity = bmw_i3.get_battery_capacity('normal')
    print(normal_capacity)
    current_lvl = 40
    soc_1 = soc(battery_level=current_lvl,
                battery_capacity=normal_capacity,
                battery_efficiency=100)
    print(soc_1)