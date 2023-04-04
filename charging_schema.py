import datetime


def charge_battery(charge_start_time, charge_end_time, charging_power):
    """
    Simulates charging a battery from charge_start_time to charge_end_time
    at a given charging_power level.
    """
    print(f"Starting battery charge at {charge_start_time}.")
    print(f"Charging at {charging_power} kW.")
    print(f"Estimated time to fully charge: {(charge_end_time - charge_start_time).total_seconds() / 3600} hours.")
    # Simulate charging process
    time_elapsed = datetime.timedelta()
    while time_elapsed < (charge_end_time - charge_start_time):
        # Charge battery at given power level
        time_elapsed += datetime.timedelta(minutes=10)
    print(f"Charging complete at {charge_end_time}.")


def charging_schema(charge_start_time, charge_end_time, charging_power, uncontrolled=False, flat=False, off_peak=False, peak_start_time=None, peak_end_time=None):
    """
    Determines the appropriate charging schema based on the input parameters and
    returns the charging start and end times for the battery.
    """
    if uncontrolled:
        # Charge at maximum power level without regard for time of day
        return charge_start_time, charge_end_time
    elif flat:
        # Charge at a constant power level throughout the charging period
        return charge_start_time, charge_end_time
    elif off_peak:
        # Charge at a constant power level during off-peak hours
        if peak_start_time is None or peak_end_time is None:
            raise ValueError("Peak start and end times must be provided for off-peak charging schema.")
        if charge_start_time.time() >= peak_end_time or charge_end_time.time() <= peak_start_time:
            # Charge entirely during off-peak hours
            return charge_start_time, charge_end_time
        else:
            # Charge partially during peak hours
            if charge_start_time.time() < peak_start_time:
                off_peak_start_time = charge_start_time
                peak_start_time = datetime.datetime.combine(charge_start_time.date(), peak_start_time)
            else:
                peak_start_time = datetime.datetime.combine(charge_start_time.date(), peak_start_time)
                off_peak_start_time = peak_end_time
            if charge_end_time.time() > peak_end_time:
                off_peak_end_time = charge_end_time
                peak_end_time = datetime.datetime.combine(charge_end_time.date(), peak_end_time)
            else:
                peak_end_time = datetime.datetime.combine(charge_end_time.date(), peak_end_time)
                off_peak_end_time = peak_start_time
            off_peak_duration = off_peak_end_time - off_peak_start_time
            off_peak_power = charging_power * (off_peak_duration.total_seconds() / 3600)
            peak_duration = peak_end_time - peak_start_time
            peak_power = charging_power * (peak_duration.total_seconds() / 3600)
            print(f"Charging {off_peak_duration} hours at {off_peak_power} kW during off-peak hours.")
            print(f"Charging {peak_duration} hours at {peak_power} kW during peak hours.")
            return off_peak_start_time, off_peak_end_time
    else:
        raise ValueError("No charging schema selected.")
