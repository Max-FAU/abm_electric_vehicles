# Initialization of the model
# Retrieve the EV Agent data
# create electric vehicle with battery
# assign battery capacity
# initialize electric vehicle


def init_car_agent(name, battery_capacity=None):
    # battery capacity for car name, min_value, expected_value, max_value in kWh
    car_dict = {
        "dummy": {
            "min_value": 30,
            "expected_value": 50,
            "max_value": 100
        },
        "bmw": {
            "min_value": 50,
            "expected_value": 70,
            "max_value": 120
        }
    }

    return choose_value(name, car_dict, battery_capacity)


def choose_value(name, input_dict, input_value):
    value_key = {
        "min": "min_value",
        "expected": "expected_value",
        "max": "max_value"
    }

    if input_value in value_key:
        return input_dict[name][value_key[input_value]]
    return input_dict[name]


if __name__ == '__main__':
    # get car variables for expected scenario
    car_expected_value = init_car_agent(name='bmw',
                                        battery_capacity='expected')
    print(car_expected_value)