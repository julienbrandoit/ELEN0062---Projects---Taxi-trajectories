import pandas as pd
import numpy as np
import sklearn
print("... DATA LOADING ...")

file_path = '../data/train.csv'
data = pd.read_csv(file_path, index_col="TRIP_ID")

print("... DATA LOADED !")

SIZE = 15000
L_PADDING = 225
f = 0.3

def haversine(pred, gt):
    """
    Havarsine distance between two points on the Earth surface.
    Parameters
    -----
    pred: numpy array of shape (N, 2)
        Contains predicted (LATITUDE, LONGITUDE).
    gt: numpy array of shape (N, 2)
        Contains ground-truth (LATITUDE, LONGITUDE).
    Returns
    ------
    numpy array of shape (N,)
        Contains haversine distance between predictions
        and ground truth.
    """
    pred_lat = np.radians(pred[:, 0])
    pred_long = np.radians(pred[:, 1])
    gt_lat = np.radians(gt[:, 0])
    gt_long = np.radians(gt[:, 1])
    dlat = gt_lat - pred_lat
    dlon = gt_long - pred_long
    a = np.sin(dlat/2)**2 + np.cos(pred_lat) * np.cos(gt_lat) * np.sin(dlon/2)**2
    d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return d
first_time = True
category_mapping_origin_call = {}
category_mapping_taxi_ID = {}
def process(S):
    global first_time
    global category_mapping_origin_call
    global category_mapping_taxi_ID
    s = data.sample(S, replace=True)
    s['POLYLINE'] = s['POLYLINE'].apply(eval)
    s.drop(s[s['POLYLINE'].apply(len) < 2].index, inplace=True)
    s['y1'] = s['POLYLINE'].apply(lambda x: x[-1][0])
    s['y2'] = s['POLYLINE'].apply(lambda x: x[-1][1])
    
    s['x'] = s['POLYLINE'].apply(lambda x: x[:min(np.random.randint(1, len(x)), L_PADDING)]) 
    s['x1'] = s['x'].apply(lambda x: [p[0] for p in x]) 
    s['x2'] = s['x'].apply(lambda x: [p[1] for p in x])
    s['L'] = s['x'].apply(len)
    s['x1'] = s['x1'].apply(lambda x: np.pad(x, (L_PADDING - len(x),0)))
    s['x2'] = s['x2'].apply(lambda x: np.pad(x, (L_PADDING - len(x),0)))
    # Create new columns with the padded sequences
    new_columns_x1 = ['x1_' + str(i) for i in range(L_PADDING)]
    new_columns_x2 = ['x2_' + str(i) for i in range(L_PADDING)]
    s = pd.concat([s, pd.DataFrame(s['x1'].tolist(), columns=new_columns_x1, index=s.index)], axis=1)
    s = pd.concat([s, pd.DataFrame(s['x2'].tolist(), columns=new_columns_x2, index=s.index)], axis=1)
    s.drop(['x1', 'x2'], axis=1, inplace=True)
    s['month'] = s['TIMESTAMP'].apply(lambda x: pd.Timestamp(x, unit='s').month)
    s['day'] = s['TIMESTAMP'].apply(lambda x: pd.Timestamp(x, unit='s').day)
    s['hour'] = s['TIMESTAMP'].apply(lambda x: pd.Timestamp(x, unit='s').hour)
    s.drop(['TIMESTAMP'], axis=1, inplace=True)
    s.drop(['POLYLINE'], axis=1, inplace=True)
    s.drop(['MISSING_DATA'], axis=1, inplace=True)
    s.drop(['x'], axis=1, inplace=True)
    s['CALL_TYPE'] = s['CALL_TYPE'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2)
    s['DAY_TYPE'] = s['DAY_TYPE'].apply(lambda x: 0 if x == 'A' else 1 if x == 'B' else 2)
    # Create a dictionary to map rare categories and other categories to integer labels
    if first_time:
        first_time = False
        # Calculate the frequency of each category in the 'ORIGIN_CALL' column
        category_frequencies_call = s['ORIGIN_CALL'].value_counts(normalize=True)
        category_frequencies_taxi = s['TAXI_ID'].value_counts(normalize=True)
        # Set a threshold for considering categories as rare
        #threshold = 0.01
        #rare_categories = category_frequencies[category_frequencies < threshold].index
        top_categories_call = category_frequencies_call.nlargest(255).index
        top_categories_taxi = category_frequencies_taxi.nlargest(255).index
        category_mapping_origin_call = {category: i for i, category in enumerate(top_categories_call)}
        category_mapping_taxi_ID = {category: i for i, category in enumerate(top_categories_taxi)}
    # Apply the mapping to the training set
    s['ORIGIN_CALL'] = s['ORIGIN_CALL'].apply(lambda x : category_mapping_origin_call.get(x, None))
    s['TAXI_ID'] = s['TAXI_ID'].apply(lambda x : category_mapping_taxi_ID.get(x, None))
    """
    s.drop(['CALL_TYPE'], axis=1, inplace=True)
    s.drop(['DAY_TYPE'], axis=1, inplace=True)
    s.drop(['ORIGIN_CALL'], axis=1, inplace=True)
    s.drop(['ORIGIN_STAND'], axis=1, inplace=True)
    """
    return s
print("... DATA PROCESSING ...")
dataset = pd.DataFrame()
while len(dataset) < SIZE:
    print(f"Filling : {(len(dataset)/SIZE * 100):.2f}%")
    dataset = pd.concat([dataset, process(SIZE - len(dataset))])
print("... DATA PROCESSED !")
Y_set = dataset[['y1', 'y2']]
X_set = dataset.drop(['y1','y2'], axis=1)
print("... DATA SPLITING ...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_set, Y_set, test_size=f)
print("... DATA SPLITED ...")
print("... GENERATING MODELS ...")
from sklearn.ensemble import HistGradientBoostingRegressor
lat_estimator = HistGradientBoostingRegressor(loss='squared_error',
                                            learning_rate=0.005,
                                            max_iter=1000, #increase this later ! 
                                            max_depth=None, #really ?
                                            min_samples_leaf=20,
                                            max_leaf_nodes=100, 
                                            verbose=1,
                                            categorical_features=['CALL_TYPE', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'month', 'day'])
long_estimator = HistGradientBoostingRegressor(loss='squared_error',
                                            learning_rate=0.005,
                                            max_iter=1000, #increase this later ! 
                                            max_depth=None, #really ?
                                            min_samples_leaf=20,
                                            max_leaf_nodes=100, 
                                            verbose=1,
                                            categorical_features=['CALL_TYPE', 'DAY_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'month', 'day'])
from sklearn.base import BaseEstimator, RegressorMixin
class CustomScorerWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, lat_estimator, long_estimator):
        self.lat_estimator = lat_estimator
        self.long_estimator = long_estimator
    def fit(self, X, y):
        self.lat_estimator.fit(X, y['y2'])
        self.long_estimator.fit(X, y['y1'])
        return self
    def predict(self, X):
        lat_predictions = self.lat_estimator.predict(X)
        long_predictions = self.long_estimator.predict(X)
        return np.column_stack((long_predictions, lat_predictions))
    def score(self, X, y, sample_weight=None):
        # Assuming y is a DataFrame with 'y1' and 'y2' columns
        y_true = y.values
        y_pred = self.predict(X)
        score = haversine(y_pred, y_true)
        # Return the negative combined error as the score (minimization problem)
        return score
    def features_importance(self):
        return self.lat_estimator.feature_importances_, self.long_estimator.feature_importances_
custom_scorer_wrapper = CustomScorerWrapper(lat_estimator, long_estimator)
custom_scorer_wrapper.fit(X_train, y_train)
print("... MODELS GENERATED !")
print("... PREDICTING ...")
score = custom_scorer_wrapper.score(X_test, y_test)

import matplotlib.pyplot as plt
plt.hist(score, bins=100)
plt.axvline(x=np.mean(score), color='r')
print("SCORE:",np.mean(score))
plt.show()

print("... PREDICTED !")