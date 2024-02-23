import pandas as pd
import numpy as np
import sklearn

file_path = '../data/train.csv'
data = pd.read_csv(file_path, index_col="TRIP_ID")

SIZE = 20000
L_PADDING = 200

train = pd.DataFrame()
test = pd.DataFrame()

def process(S):
    s = data.sample(S, replace=True)
    s['POLYLINE'] = s['POLYLINE'].apply(eval)
    s.drop(s[s['POLYLINE'].apply(len) < 2].index, inplace=True)

    s['y1'] = s['POLYLINE'].apply(lambda x: x[-1][0])
    s['y2'] = s['POLYLINE'].apply(lambda x: x[-1][1])
    
    s['x'] = s['POLYLINE'].apply(lambda x: x[:np.random.randint(1, min(len(x), L_PADDING))]) 
    s['x1'] = s['x'].apply(lambda x: [p[0] for p in x]) 
    s['x2'] = s['x'].apply(lambda x: [p[1] for p in x]) 

    # Pad sequences with zeros to the maximum length
    s['x1'] = s['x1'].apply(lambda x: np.pad(x, (L_PADDING - len(x),0)))
    s['x2'] = s['x2'].apply(lambda x: np.pad(x, (L_PADDING - len(x),0)))

    # Create new columns with the padded sequences
    s[['x1_' + str(i) for i in range(L_PADDING)]] = pd.DataFrame(s['x1'].tolist(), index=s.index)
    s[['x2_' + str(i) for i in range(L_PADDING)]] = pd.DataFrame(s['x2'].tolist(), index=s.index)
    s.drop(['x1', 'x2'], axis=1, inplace=True)

    s.drop(['POLYLINE'], axis=1, inplace=True)
    s.drop(['MISSING_DATA'], axis=1, inplace=True)
    s.drop(['x'], axis=1, inplace=True)

    s.drop(['CALL_TYPE'], axis=1, inplace=True)
    s.drop(['DAY_TYPE'], axis=1, inplace=True)
    s.drop(['ORIGIN_CALL'], axis=1, inplace=True)
    s.drop(['ORIGIN_STAND'], axis=1, inplace=True)

    return s

while len(train) < SIZE:
    train = pd.concat([train, process(SIZE - len(train))])

while len(test) < SIZE/2:
    test = pd.concat([test, process(int(SIZE/2) - len(test))])

y_train = train[['y1', 'y2']]
X_train = train.drop(['y1','y2'], axis=1)
y_test = test[['y1', 'y2']]
X_test = test.drop(['y1','y2'], axis=1)

from sklearn.multioutput import MultiOutputRegressor
from sklearn import linear_model

lasso = MultiOutputRegressor(linear_model.Lasso(alpha=0.1, fit_intercept=True), n_jobs=6)
lasso.fit(X_train, y_train)

y_ls_pred_lasso = lasso.predict(X_train)
y_ts_pred_lasso = lasso.predict(X_test)

ridge = MultiOutputRegressor(linear_model.Ridge(alpha=0.1, fit_intercept=True), n_jobs=6)
ridge.fit(X_train, y_train)

y_ls_pred_ridge = ridge.predict(X_train)
y_ts_pred_ridge = ridge.predict(X_test)


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

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.title("LASSO")
plt.hist(haversine(y_ls_pred_lasso, y_train.values), bins=100, label='train', alpha=0.5, density=True)
plt.hist(haversine(y_ts_pred_lasso, y_test.values), bins=100, label='test', alpha=0.5, density=True)
print("LASSO : LS:",np.mean(haversine(y_ls_pred_lasso, y_train.values)),"TS:",np.mean(haversine(y_ts_pred_lasso, y_test.values)))
plt.legend()
plt.xlim(0, 10)
plt.subplot(1,2,2)
plt.title("RIDGE")
plt.hist(haversine(y_ls_pred_ridge, y_train.values), bins=100, label='train', alpha=0.5, density=True)
plt.hist(haversine(y_ts_pred_ridge, y_test.values), bins=100, label='test', alpha=0.5, density=True)
print("RIDGE : LS:",np.mean(haversine(y_ls_pred_ridge, y_train.values)),"TS:",np.mean(haversine(y_ts_pred_ridge, y_test.values)))
plt.legend()
plt.xlim(0, 10)
plt.show()
