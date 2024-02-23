from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Chunk size for incremental computation
chunk_size = 1e5
# Load the training data
file_path = '../data/full_train_data_coarse.csv'
chunksize = 1e5
data = pd.read_csv(file_path, chunksize=chunksize)
SIZE = 1710670
NBR_INIT_FEATURES = 8 - 1

# Initialize an array to store mutual information scores
mutual_info_scores = np.zeros(NBR_INIT_FEATURES)

print("START PROCESSING")
# Iterate over chunks of the training data
for i, c in enumerate(data):
    y_chunk = c['POLYLINE_Y']
    X_chunk = c.drop(columns=['POLYLINE_Y'])

    print(f"Chunk {i} processed")
    # Compute mutual information for each feature in the chunk
    chunk_mutual_info = mutual_info_regression(X_chunk, y_chunk, discrete_features='auto')
    # Aggregate the results
    mutual_info_scores += chunk_mutual_info
    
# Average mutual information scores over all chunks
#mutual_info_scores /= (len(X_train) // chunk_size)

# Sort features based on mutual information scores
sorted_features = np.argsort(mutual_info_scores)[::-1]
print(sorted_features)
print("END PROCESSING")