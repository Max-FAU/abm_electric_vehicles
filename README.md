# abm_electric_vehicles

## Installation
To start the project navigate to an existing virtual environment or create a new one  
1. Active the virtual environment you want to work in  
2. Navigate to the directory where the requirements.txt file is located  
3. Run in terminal following command:  
`pip install -r requirements.txt`  
4. Open the file `project_paths.py` and edit the MOBILITY_DATA_DIRECTORY_PATH to the path with mobility_profiles
## Data
### Input Files
#### car_values.json
Holding data for the 10 most popular electric vehicles in Germany.<br>
Number of vehicles is taken from Kraftfahrt-Bundesamt<br>
https://www.kba.de/DE/Statistik/Fahrzeuge/Neuzulassungen/neuzulassungen_node.html<br>

Charging power and battery capacities are taken from ADAC<br>
https://www.adac.de/rund-ums-fahrzeug/autokatalog/marken-modelle/<br>

File is structured as follows:<br>
"car_model_name": {<br>
    "battery_capacity": 77,<br>
    "charging_power_ac": 11,<br>
    "charging_power_dc": 50,<br>
    "number": 30000<br>
}<br>

#### power_station_values.json
Having a dict with three different charging values, possible to be delivered by charging stations. <br>
Depending on the location they have a different power they could charge with.

#### h0_profile.csv
Represents a standard load profile of a private customer.<br>
The csv file is taken from BDEW<br>
https://www.bdew.de/energie/standardlastprofile-strom/<br>

#### single_phase_transformers.json
Has transformer values in kVA representing standard single phase transformer sizes.<br>
Data can be found at Schneider Electric<br>
https://www.se.com/us/en/download/document/7400HO9501/<br>

#### private_cars.json
The mobility data has been labeled in previous work according to the driving profiles.<br>
File is storing car ids of private cars.<br>

#### Mobility Data
Mobility data files should have the following columns, named like this:<br>
'TIMESTAMP', 'TRIPNUMBER', 'DELTAPOS', 'CLUSTER', 'ECONSUMPTION', 'ID_PANELSESSION', 'ID_TERMINAL'<br>
TIMESTAMP: The date and time of the measurement (day/month/year hour.minute.second)<br>
TRIPNUMBER: This is an increasing number starting from 0 for the first trip<br>
DELTAPOS: Is the position in meters the car has travelled since last measurement<br>
CLUSTER: Depending on the location this column holds a 1 for home 2 for work or a 0 for everywhere else<br>
ECONSUMPTION: Is holding the consumption the car had since the last measurement in kWh<br>
ID_PANELSESSION: Is holding a value for each state of the electric vehicle 0=ignition, 1=driving, 2=engine turn-off<br>
ID_TERMINAL: The ID of the vehicle<br>

## Usage
### Run in IDE
Open `main.py` after setting file paths. <br>
Define your scenario to simulate. <br>

start_date = '2008-07-13' <br>
end_date = '2008-07-20' <br>
model_runs = 1 <br>

num_cars_normal = 1 <br>
num_cars_off_peak = 0 <br>
num_transformers = 1 <br>
num_customers = 1 <br>
car_target_soc = 100 <br>
car_charging_algo = False <br>

### Run in command line terminal
To run the simulation in your command line terminal. <br>
Open in windows `cmd` <br>
Navigate to the directory where `main.py` file is located. <br>
Use your python interpreter in your virtual environment to have all packages. <br>

Run following command (created a virtual environment in PycharmProjects\mesa):<br>
`C:\Users\USERNAME\PycharmProjects\mesa\venv\Scripts\python.exe main.py --num_cars_normal 100 --num_cars_off_peak 0 --num_transformers 1 --num_customers 100`

Note you have following args to pass:<br>
`--num_cars_normal`<br>
`--num_cars_off_peak`<br>
`--num_transformers`<br>
`--num_customers`<br>


## Contribution
This was part of the master thesis written @ Bits 2 Energy Lab University of Erlangen Nuremberg.

## Credits

## License
Free to use!

## Known Issues
- The charging efficiency is currently not implemented correctly
- Calculating the charging time at the end of simulation does not update correctly
- Peak load for transformer sizing should be set to yearly maximum peak load

## Contact
maximilian.brueggemann@fau.de

## Results
![Alt text](abm_electric_vehicles/data_analytics/Scenario 1_weekly_profile_total.png?raw=true "Generated Weekly Load Profile")
