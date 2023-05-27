import math


class Transformer:
    def __init__(self,
                 num_households):
        """
        Class to implement transformers and to calculate the maximum capacity of the transformer.
        :param num_households: Number of agents connected to the transformer
        """
        self.num_households = num_households  # num_households == num EV Agents
        self.power_household = None

        # calculation of transformer capacity
        self.customers_contracted_power = []
        self.f_safety = 1.5  # long lifetime of transformer means high safety
        self.p_over = 10
        self.capacity = None

    def set_power_household(self):
        volt = 230
        ampere = 63
        phases = 1
        self.power_household = volt * ampere * phases
        # 14490 watt

    def set_customers_contracted_power(self):
        for i in range(self.num_households):
            single_customer_value = self.power_household / 1000  # in kW
            self.customers_contracted_power.append(single_customer_value)

    def get_c_diversity(self):
        """Set diversity factor, used if load profiles created artificially."""
        return 0.2 + 0.8 / math.sqrt(self.num_households)

    def set_transformer_power_capacity(self):
        """
        p_pt = transformer power capacity
        c_diversity = 0.2 + 0.8 / sqrt(n)   # this is only valid for private customers
        c_diverstiy = 0.5 + 0.5 / sqrt(n)
        n = expected number of customers
        f_safety = safety margin to account for the power factor and for future load growth
        c_diversity = diversity (or simultaneity) coefficient
        p_over = oversized power capacity defined by the standard size of the transformer
        https://www.sciencedirect.com/science/article/pii/S0960148117310649?via%3Dihub

        """
        self.capacity = self.get_c_diversity() * sum(self.customers_contracted_power) * self.f_safety + self.p_over
        # print("Transformer with a capacity of {} kW".format(self.capacity))

    def initialize_transformer(self):
        self.set_power_household()
        self.set_customers_contracted_power()
        self.set_transformer_power_capacity()

    def get_max_capacity(self):
        return self.capacity


if __name__ == '__main__':
    # We could take one of these transformers, e.g. ABB DRY-TYPE TRANSFORMER 25 kVA 480-120/240V
    # https://electrification.us.abb.com/products/transformers-low-voltage-dry-type
    # take siemens https://mall.industry.siemens.com/mall/de/WW/Catalog/Products/10283675
    # size depends on the phases we want
    # usually we have as output 400 V

    transformer = Transformer(num_households=100)
    transformer.initialize_transformer()
    capacity = transformer.get_max_capacity()
    print(capacity)
