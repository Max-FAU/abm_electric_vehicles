import pandas as pd
import matplotlib.pyplot as plt
import random
import plotly.express as px
import numpy as np


def replace_location_panelsession(mobility_data):
    location_type = {0: "ignition", 1: "driving", 2: "engine turn-off"}
    mobility_data['id_locationtype'] = mobility_data['id_locationtype'].replace(location_type)
    panel_session = {0: "urban", 1: "highway", 2: "extra urban"}
    mobility_data['id_panelsession'] = mobility_data['id_panelsession'].replace(panel_session)
    return mobility_data


# create df with data for only one trip
def create_df_one_trip(mobility_data, trip_no, plot=False):
    mobility_data = mobility_data[(mobility_data['TRIPNUMBER'] == trip_no)]

    mobility_data = mobility_data[['TRIPNUMBER',
                                   'TIMESTAMP',
                                   'ID_LOCATIONTYPE',
                                   'ID_PANELSESSION',
                                   'LONGITUDE',
                                   'LATITUDE']]

    column_names = ['trip',
                    'start',
                    'id_locationtype',
                    'id_panelsession',
                    'lon',
                    'lat']

    mobility_data.columns = column_names

    mobility_data = replace_location_panelsession(mobility_data)

    if plot:
        color_scale = [(0, 'black'), (1, 'red')]
        fig = px.scatter_mapbox(mobility_data,
                                lat="lat",
                                lon="lon",
                                hover_name="trip",
                                hover_data=["trip", "start", "id_locationtype", "id_panelsession"],
                                color="trip",
                                color_continuous_scale=color_scale,
                                # size="Listed",
                                zoom=8,
                                height=800,
                                width=800)

        fig.update_layout(mapbox_style="open-street-map", coloraxis_showscale=False)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.show()

    return mobility_data


def create_df(start=False, end=False, mobility_data=None, test=True):
    if start and end:
        return print('error')
    if start:
        loc = 0
        end = False
        column_name = ['start', 'id_locationtype', 'id_panelsession', 'lon', 'lat']
    if end:
        loc = -1
        # loop over each
        column_name = ['end', 'id_locationtype', 'id_panelsession', 'lon', 'lat']

    # pick 10 random trips
    unique_list = mobility_data['TRIPNUMBER'].unique()
    if test:
        unique_list = [849, 850, 851, 852, 853, 854, 855, 856]

    trips = {}
    for trip in unique_list:
        # create dict with start and end points of trips for one single car
        trips[trip] = (mobility_data[(mobility_data['TRIPNUMBER'] == trip)].iloc[loc]['TIMESTAMP'],
                       # mobility_data[(mobility_data['TRIPNUMBER'] == trip)].iloc[-1]['TIMESTAMP'],
                       mobility_data[(mobility_data['TRIPNUMBER'] == trip)].iloc[loc]['ID_LOCATIONTYPE'],
                       mobility_data[(mobility_data['TRIPNUMBER'] == trip)].iloc[loc]['ID_PANELSESSION'],
                       mobility_data[(mobility_data['TRIPNUMBER'] == trip)].iloc[loc]['LONGITUDE'],
                       mobility_data[(mobility_data['TRIPNUMBER'] == trip)].iloc[loc]['LATITUDE']
                       )

    # transpose columns
    trips = pd.DataFrame(trips).T
    # set real column names

    trips.columns = column_name
    trips['trip'] = trips.index

    if start:
        # create timestamps for start and end points
        trips['start'] = pd.to_datetime(trips['start'])
    else:
        trips['end'] = pd.to_datetime(trips['end'])

    trips = replace_location_panelsession(trips)
    return trips


def print_start_endpoints_trips(start, end):
    color_scale = [(0, 'black'), (1, 'red')]
    fig = px.scatter_mapbox(start,
                            lat="lat",
                            lon="lon",
                            hover_name="trip",
                            hover_data=["trip", "start", "id_locationtype", "id_panelsession"],
                            color="trip",
                            color_continuous_scale=color_scale,
                            # size="Listed",
                            zoom=8,
                            height=800,
                            width=800)

    fig2 = px.scatter_mapbox(end,
                            lat="lat",
                            lon="lon",
                            hover_name="trip",
                            hover_data=["trip", "end", "id_locationtype", "id_panelsession"],
                            color="trip",
                            color_continuous_scale=color_scale,
                            # size="Listed",
                            zoom=8,
                            height=800,
                            width=800)

    fig.add_trace(fig2.data[0])

    fig.update_layout(mapbox_style="open-street-map", coloraxis_showscale=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


if __name__ == "__main__":
    # cars_overview = r"C:\Users\Max\Desktop\Master Thesis\Data\cars_measurement_start_end.xlsx"
    # cars = pd.read_excel(cars_overview)
    mobility_data = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\quarterly_simulation_80.csv"
    # home = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\Parking_Cluster_1_LatLongMODES_Home.csv"
    # work = r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\Parking_Cluster_2_LatLongMODES_Work.csv"
    # home = pd.read_csv(home)
    mobility_data = pd.read_csv(mobility_data)
    start = create_df(start=True, mobility_data=mobility_data, test=False)
    end = create_df(end=True, mobility_data=mobility_data, test=False)
    # create_df_one_trip(mobility_data=mobility_data, trip_no=100, plot=True)


    # can print and generate plot with all start and end points for the car and the trips of the car
    print_start_endpoints_trips(start=start, end=end)



