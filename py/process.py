import pandas as pd
import numpy as np
import os

SEED = 42
np.random.seed(SEED)

# Load the training data
file_path = '../data/train.csv'
chunksize = 1e5
data = pd.read_csv(file_path, index_col="TRIP_ID", chunksize=chunksize)
SIZE = 1710670

def process(sub_data):
    # Filter rows with 'MISSING_DATA' set to False
    #sub_data = sub_data[sub_data['MISSING_DATA'] == False]

    # Drop the 'MISSING_DATA' column
    sub_data = sub_data.drop(columns=['MISSING_DATA'])
    
    # Replace values in 'CALL_TYPE' and 'DAY_TYPE' columns
    sub_data['CALL_TYPE'] = sub_data['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])
    sub_data['DAY_TYPE'] = sub_data['DAY_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])
    
    # Filter rows with 'POLYLINE' not equal to '[]'
    sub_data = sub_data[sub_data['POLYLINE'] != '[]']

    # Convert 'POLYLINE' to list of coordinates
    sub_data['POLYLINE'] = sub_data['POLYLINE'].apply(eval)

    # Function to keep only the first r elements for each subtable
    def keep_random_elements(tab):
        if(len(tab) < 2):
            return None, None
        r = np.random.randint(1, len(tab))
        return tab[:r], tab[-1]

    # Apply the 'keep_random_elements' function to 'POLYLINE' column
    sub_data[['POLYLINE_X', 'POLYLINE_Y']] = sub_data['POLYLINE'].apply(keep_random_elements).apply(pd.Series)

    # Drop the 'POLYLINE' column
    sub_data = sub_data.drop(columns=['POLYLINE'])
    sub_data = sub_data[sub_data['POLYLINE_X'] != None]

    return sub_data

# Assuming data is a list of DataFrames
file_path_train = '../data/full_train_data_coarse.csv'

# Remove existing files
if os.path.exists(file_path_train):
    os.remove(file_path_train)

header_written = False  # To write header only once

s = 0
print("START PROCESSING")
for i, c in enumerate(data):
    s += len(c)
    train_data = c
    train_data = process(train_data) 
    train_data.to_csv(file_path_train, mode='a', header=not header_written, index=False)
    
    # Update header_written after writing the header
    if not header_written:
        header_written = True

    print(f"about {float(s/SIZE) * 100} % processed")