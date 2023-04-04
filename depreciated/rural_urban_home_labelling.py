import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Check if home is in polygon -> If TRUE -> Label -> Return -> IF None -> Label = 'Not found' RETURN
home_base = pd.read_csv(r"C:\Users\Max\Desktop\Master Thesis\Data\MobilityProfiles_EV_Data\Parking_Cluster_1_LatLongMODES_Home.csv")
# keep only first 3 columns
home_locations = home_base.iloc[:, :3]

# get labels of different NUTS3
labels = pd.read_csv(r"C:\Users\Max\Desktop\Master Thesis\Data\rural_labels_EU_data_NUTS2021.csv", sep=";")
labels = labels[labels['NUTS_ID'].str.startswith('IT')].reset_index(drop=True)
labels = labels.iloc[:, :3]

# home base (car location) -> labels (rural, urban, suburban) -> shape file (to get regions)
shapefile = gpd.read_file(r"C:\Users\Max\Desktop\Master Thesis\Data\NUTS_RG_20M_2021\NUTS_RG_20M_2021_3035.shp")
# map to correct coordinates
shapefile = shapefile.to_crs('epsg:4326')
# keep only relevant labels for italian NUTS 3
list_labels = list(labels['NUTS_ID'])
shapefile = shapefile[shapefile['NUTS_ID'].isin(list_labels)]

# create a dataframe with relevant NUTS and geometry shapes to check if point is in shape
df_poly = shapefile[['NUTS_ID', 'geometry']]
df_poly_dict = dict(zip(df_poly.NUTS_ID, df_poly.geometry))

# Create points list to compare if they are within any of the polygons
points = list(zip(home_locations['Mode Long 1'], home_locations['Mode Lat 1']))
points = dict(zip(home_locations.ID, points))

# points = list(zip(home_locations['Mode Lat 1'] * 100000, home_locations['Mode Long 1'] * 100000))

# create list of points, points are the home locations of the cars
_car_ids = []
_pnts = []
for car_id, car_geom in points.items():
    _car_ids += [car_id]
    _pnts += [Point(car_geom)]

point_dict = {'car_id': _car_ids, 'geometry': _pnts}
# create geometry shapes out of points
pnts = gpd.GeoDataFrame(point_dict)

# create empty dict
labelled_cars = dict()

# loop over the cars with the corresponding home locations
for car_id, point_geo in pnts.itertuples(index=False):
    # loop over polygons and shapes
    for key, poly_geom in df_poly_dict.items():
        # check if point is within shape
        if point_geo.within(poly_geom):
            labelled_cars[car_id] = key

            # # check if key is already in labelled_cars
            # if key in labelled_cars:
            #     # append car_id to the list of entries for the key
            #     labelled_cars[key].append(car_id)
            # else:
            #     # create new key value pair for dict
            #     labelled_cars[key] = [car_id]


# Only 894 cars are labelled -> Maybe enough
# label the cars with 1, 2, 3
labelled_cars_df = pd.DataFrame(labelled_cars.items(), columns=['CAR_ID', 'NUTS_ID'])
labelled_cars_df = labelled_cars_df.merge(labels, left_on='NUTS_ID', right_on='NUTS_ID')
labelled_cars_df.to_csv("labelled_cars.csv")
print(labelled_cars_df.head(5))