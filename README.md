# abm_electric_vehicles

## Installation
To start the project navigate to an existing virtual environment or create a new one  
1. Active the virtual environment you want to work in  
2. Navigate to the directory where the requirements.txt file is located  
3. Run in terminal following command:  
`pip install -r requirements.txt`  
4. Open the file `project_paths.py` and edit the MOBILITY_DATA_DIRECTORY_PATH to the path with mobility_profiles
## Usage
### Input Files
#### car_values.json
Holding data for the 10 most popular electric vehicles in Germany.
Number of vehicles is taken from Kraftfahrt-Bundesamt
https://www.kba.de/DE/Statistik/Fahrzeuge/Neuzulassungen/neuzulassungen_node.html

Charging power and battery capacities are taken from ADAC
https://www.adac.de/rund-ums-fahrzeug/autokatalog/marken-modelle/

File is structured as follows:
"car_model_name": {
    "battery_capacity": 77,
    "charging_power_ac": 11,
    "charging_power_dc": 50,
    "number": 30000
}

#### h0_profile.csv
Represents a standard load profile of a private customer.
The csv file is taken from BDEW
https://www.bdew.de/energie/standardlastprofile-strom/

#### single_phase_transformers.json
Has transformer values in kVA representing standard single phase transformer sizes.
Data can be found at Schneider Electric
https://www.se.com/us/en/download/document/7400HO9501/

#### private_cars.json
The mobility data has been labeled in previous work according to the driving profiles.
File is storing car ids of private cars.

#### Mobility Data
Mobility data files should have the following columns, named like this:
'TIMESTAMP', 'TRIPNUMBER', 'DELTAPOS', 'CLUSTER', 'ECONSUMPTION', 'ID_PANELSESSION', 'ID_TERMINAL'
TIMESTAMP: The date and time of the measurement (day/month/year hour.minute.second)
TRIPNUMBER: This is an increasing number starting from 0 for the first trip
DELTAPOS: Is the position in meters the car has travelled since last measurement
CLUSTER: Depending on the location this column holds a 1 for home 2 for work or a 0 for everywhere else
ECONSUMPTION: Is holding the consumption the car had since the last measurement in kWh
ID_PANELSESSION: Is holding a value for each state of the electric vehicle 0=ignition, 1=driving, 2=engine turn-off
ID_TERMINAL: The ID of the vehicle

## Contribution

## Credits

## License

## Contact
