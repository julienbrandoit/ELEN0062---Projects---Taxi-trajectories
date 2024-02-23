import pandas as pd
import numpy as np


file_path = '../data/train.csv'
data = pd.read_csv(file_path, index_col="TRIP_ID")

data = data[:1000]
print(len(data))