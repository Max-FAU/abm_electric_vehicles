import json
from customer_agent import PowerCustomer
from mesa import Agent


class Transformer(Agent):
    def __init__(self,
                 unique_id,
                 model,
                 num_households,
                 peak_load,
                 f_safety=1.2,
                 power_factor=0.9):
        """
        Class to implement transformers and to calculate the maximum capacity of the transformer.
        :param num_households: Number of agents connected to the transformer
        """
        super().__init__(unique_id, model)
        self.type = 'Transformer'
        self.unique_id = unique_id

        self.num_households = num_households  # num_households == num EV Agents
        self.peak_load = peak_load
        self.f_safety = f_safety
        self.power_factor = power_factor

        self._timestamp = None

        self.total_peak_load = None
        self.p_over = None
        self.capacity_kva_interim = None
        self.capacity_kva = None
        self.capacity_kw = None
        self.initialize_transformer()

    @staticmethod
    def possible_size():
        # https://www.se.com/us/en/download/document/7400HO9501/
        with open('input/single_phase_transformers.json') as f:
            data = json.load(f)
        return data['kVA']

    def get_peak_load(self):
        return self.peak_load

    def get_num_households(self):
        return self.num_households

    def set_total_peak_load(self):
        num = self.get_num_households()
        peak = self.get_peak_load()
        self.total_peak_load = num * peak

    def get_total_peak_load(self):
        return self.total_peak_load

    def set_capacity_kva_interim(self):
        total_peak = self.get_total_peak_load()
        self.capacity_kva_interim = total_peak / self.power_factor * self.f_safety

    def get_capacity_kva_interim(self):
        return self.capacity_kva_interim

    def set_capacity_kva_value(self, value):
        self.capacity_kva = value

    def get_capacity_kva(self):
        return self.capacity_kva

    def set_capacity_kva(self):
        interim_kva = self.get_capacity_kva_interim()
        sizes = Transformer.possible_size()
        if interim_kva > 333:
            print("Consider three phase transformer sizing.")
            self.set_capacity_kva_value(333)
        else:
            possible_size = [x for x in sizes if x > interim_kva]
            self.set_capacity_kva_value(possible_size[0])

        total = self.get_capacity_kva()
        self.p_over = total - interim_kva

    def set_capacity_kw(self):
        kva = self.get_capacity_kva()
        self.capacity_kw = kva * self.power_factor

    def get_capacity_kw(self):
        return self.capacity_kw

    def initialize_transformer(self):
        self.set_total_peak_load()
        self.set_capacity_kva_interim()
        self.set_capacity_kva()
        self.set_capacity_kw()

    def get_unique_id(self):
        return self.unique_id

    def step(self):
        # No actions performed by transformer at the moment
        pass


if __name__ == '__main__':
    import pandas as pd
    start_date = pd.to_datetime('2008-07-13 00:00:00')
    end_date = pd.to_datetime('2008-07-13 23:00:00')

    customer = PowerCustomer(yearly_cons_household=3500,
                             start_date=start_date,
                             end_date=end_date)
    customer.initialize_customer()
    peak = customer.get_peak_load_kw()

    test = Transformer(num_households=20,
                       peak_load=peak)
    test.initialize_transformer()
    capacity_kva = test.get_capacity_kva()
    capacity_kw = test.get_capacity_kw()

