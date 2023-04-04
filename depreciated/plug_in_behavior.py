import datetime
import random

def plug_in_behavior(plug_in_time, plug_in_frequency):
    """
    Determines whether or not to plug in the vehicle at a given time, based on a given plug-in frequency.
    """
    if plug_in_frequency == "Whenever possible":
        # Plug in every time the vehicle is parked
        return True
    elif plug_in_frequency == "Once a week":
        # Plug in once a week at a random day and time
        plug_in_day = random.randint(0, 6)
        plug_in_time = datetime.datetime.combine(plug_in_time.date() - datetime.timedelta(days=plug_in_time.weekday()) + datetime.timedelta(days=plug_in_day), plug_in_time.time())
        return plug_in_time == plug_in_time.replace(hour=8, minute=0, second=0, microsecond=0)
    elif plug_in_frequency == "Three times a week":
        # Plug in three times a week at random days and times
        plug_in_days = random.sample(range(7), 3)
        plug_in_days.sort()
        plug_in_times = [datetime.datetime.combine(plug_in_time.date() - datetime.timedelta(days=plug_in_time.weekday()) + datetime.timedelta(days=day), datetime.time(hour=8, minute=0, second=0)) for day in plug_in_days]
        return plug_in_time in plug_in_times
    else:
        raise ValueError("Invalid plug-in frequency selected.")
