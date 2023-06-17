import auxiliary as aux
from agents.car_agent import ElectricVehicle


# TODO implement charging in flat manner, means calculate the time the car stands
# TODO divide the charging power by the time the car has to charge
class ElectricVehicleFlatCharge(ElectricVehicle):
    def __init__(self, unique_id, model, car_model, start_date, end_date, target_soc, max_transformer_capacity):
        super().__init__(unique_id, model, car_model, start_date, end_date, target_soc, max_transformer_capacity)
        self.flat_min_power = 1.22
        self.flat_max_power = 3.7

    def set_flat_charging(self):
        # TODO Not only take the current timestep, change the charging value function
        # TODO to implement in charging function more limits
        charging_power = aux.convert_kw_kwh(kwh=self.get_charging_value())
        charging_duration = self.get_charging_duration()

        flat_power = charging_power / charging_duration
        if flat_power < self.flat_min_power:
            flat_power = self.flat_min_power

        if flat_power > self.flat_max_power:
            flat_power = self.flat_max_power

        # print(flat_power)

