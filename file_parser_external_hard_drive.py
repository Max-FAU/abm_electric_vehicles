import psutil
import pandas as pd
import glob


def check_for_external_hard():
    # get a list of all disk partitions (including external drives)
    disk_partitions = psutil.disk_partitions()

    hard_drive_plugged_in = False

    # loop through all partitions and check if any of them are external drives
    for partition in disk_partitions:
        print(partition.device)
        if 'D' in partition.device:
            print("An external hard drive found!")

            hard_drive_plugged_in = True

            directory_path = 'D:\Max_Mobility_Profiles\quarterly_simulation'
            # use glob to get a list of all CSV files in the directory
            csv_files = glob.glob(directory_path + '*.csv')

            print(csv_files)
            for file in csv_files:
                if 'test' not in file.lower():
                    print(file)
                #     file_path = os.path.join(directory_path, filename)
                #     with open(file_path, 'r') as file:
                #         file_content = file.read()
                #         # process the file content here

                # create a CSV file reader object with a chunksize of 1000 rows
                # csv_reader = pd.read_csv(file, chunksize=1000)
                #
                # # loop through each chunk and process it
                # for chunk in csv_reader:
                #     # perform your simulation on the chunk here
                #     pass

    if not hard_drive_plugged_in:
        print('No external hard drive found!')

# Quality checks on timestamps
def parse_csv(filename):
    # Read in the CSV file
    df = pd.read_csv(filename)

    # Check for missing values in the timestamp column
    if df['timestamp'].isnull().values.any():
        raise ValueError('Timestamp column contains missing values.')

    # Check that the timestamp column contains only datetime values
    try:
        pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError('Timestamp column contains non-datetime values.')

    # Clean the file by dropping any rows with missing values
    df.dropna(inplace=True)

    return df


if __name__ == '__main__':
    check_for_external_hard()