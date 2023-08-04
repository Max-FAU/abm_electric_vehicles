from pathlib import Path

# Set project paths
PROJECT_DIR = Path.cwd()

INPUT_PATH = PROJECT_DIR / 'input'
# Customer agent paths
H0_FILE_PATH = INPUT_PATH / 'h0_profile.csv'
H0_CLEANED_FILE_PATH = INPUT_PATH / 'cleaned_h0_profile.csv'

CAR_VALUES_PATH = INPUT_PATH / 'car_values.json'
CHARGER_VALUE_PATH = INPUT_PATH / 'power_station_values.json'
TRANSFORMER_VALUES_PATH = INPUT_PATH / 'single_phase_transformers.json'
MEDIAN_TRIP_LEN_PATH = INPUT_PATH / 'median_trip_length.csv'

# Change this to specify where your mobility data is stored
MOBILITY_DATA_DIRECTORY_PATH = Path(r"E:\PrakharSims\Flex_TechPotential\Resolution_15min_4h\quarterly_simulation")

RESULT_PATH = PROJECT_DIR / 'results'
