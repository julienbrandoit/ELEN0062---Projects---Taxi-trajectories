import pandas as pd
import numpy as np
import math

SEED = 42
np.random.seed(SEED)

def haversine(pred, gt):
        """
        Havarsine distance between two points on the Earth surface.

        Parameters
        -----
        pred: numpy array of shape (N, 2)
            Contains predicted (LONGITUDE, LATITUDE).
        gt: numpy array of shape (N, 2)
            Contains ground-truth (LONGITUDE, LATITUDE).

        Returns
        ------
        numpy array of shape (N,)
            Contains haversine distance between predictions
            and ground truth.
        """
        pred_lat = np.radians(pred[:, 1])
        pred_long = np.radians(pred[:, 0])
        gt_lat = np.radians(gt[:, 1])
        gt_long = np.radians(gt[:, 0])

        dlat = gt_lat - pred_lat
        dlon = gt_long - pred_long

        a = np.sin(dlat/2)**2 + np.cos(pred_lat) * np.cos(gt_lat) * np.sin(dlon/2)**2

        d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return d


def get_orientation(x0,y0,xf,yf):
    
    x0_rad,y0_rad,xf_rad,yf_rad = np.radians([x0,y0,xf,yf])
    
    dx = xf_rad - x0_rad
    dy = yf_rad - y0_rad
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    angle = (angle + 360) % 360
    
    return angle

def preprocess_cut(s, validation = False):
    global lower_bound_len, upper_bound_len
    global lower_bound_y1, upper_bound_y1
    global lower_bound_y2, upper_bound_y2

    if not validation:
        s.drop(s[s['POLYLINE'].apply(lambda x :             len(x) <= 1 ## Too short to have features and output
                                                        or len(x) < lower_bound_len or len(x) > upper_bound_len ##outliers
                                                        or x[-1][0] < lower_bound_y1 or x[-1][0] > upper_bound_y1 ##outliers
                                                        or x[-1][1] < lower_bound_y2 or x[-1][1] > upper_bound_y2) == True].index, inplace=True) ##outliers
    else:
        s.drop(s[s['POLYLINE'].apply(lambda x :             len(x) <= 1)  == True].index, inplace=True)

    c = s['POLYLINE'].apply(lambda x : max(1,np.random.randint(1, max(2, len(x)))) if len(x) < 48 else np.random.randint(14, 49))

    return c

def process_set(dataset, size, test_set = False, validation=False):
    global category_mapping_origin_call
    global category_mapping_origin_stand
    global category_mapping_taxi_ID
    global k

    if test_set:
        s = dataset.copy()
    else:
        print("---- processing :", int(size), flush = True)
        size = int(size)
        temp_size = size
        batch_size = 2048
        if temp_size > batch_size:
            size = batch_size
        s = dataset.sample(size, replace=False, random_state=SEED)

    s['POLYLINE'] = s['POLYLINE'].apply(eval)

    if test_set:
        s['r'] = s['POLYLINE'].apply(lambda x : len(x))
    else:
        s['r'] = preprocess_cut(s, validation = validation)

    s['x'] = s.apply(lambda row: row['POLYLINE'][:k]+ row['POLYLINE'][row['r'] - k:row['r']] if row['r'] >=2*k 
                     else row['POLYLINE'][:row['r']//2] + [[0,0]] * (2*k - row['r']) + row['POLYLINE'][row['r'] - math.ceil(row['r']/2):row['r']], axis=1)

    s['x1'] = s['x'].apply(lambda x: [p[0] for p in x])
    s['x2'] = s['x'].apply(lambda x: [p[1] for p in x])

    # Create new columns with the padded sequences
    new_columns_x1 = ['x1_' + str(i) for i in range(2*k)]
    new_columns_x2 = ['x2_' + str(i) for i in range(2*k)]

    if not test_set:
        s.drop(s[s['x'].apply(len) < 1].index, inplace=True)


    s['y1'] = s['POLYLINE'].apply(lambda x: x[-1][0])
    s['y2'] = s['POLYLINE'].apply(lambda x: x[-1][1])

    s['orientation'] = s['x'].apply(lambda x : get_orientation(x[0][0], x[0][1], x[-1][0], x[-1][1]))

    s['x1'] = s['x'].apply(lambda x: [p[0] for p in x])
    s['x2'] = s['x'].apply(lambda x: [p[1] for p in x])

    s = pd.concat([s, pd.DataFrame(s['x1'].tolist(), columns=new_columns_x1, index=s.index)], axis=1)
    s = pd.concat([s, pd.DataFrame(s['x2'].tolist(), columns=new_columns_x2, index=s.index)], axis=1)
    s.drop(['x1', 'x2'], axis=1, inplace=True)

    s['month'] = s['TIMESTAMP'].apply(lambda x: pd.Timestamp(x, unit='s').month)
    s['day'] = s['TIMESTAMP'].apply(lambda x: pd.Timestamp(x, unit='s').day)
    s['hour'] = s['TIMESTAMP'].apply(lambda x: pd.Timestamp(x, unit='s').hour)
    s['quarter'] = s['TIMESTAMP'].apply(lambda x: (pd.Timestamp(x, unit='s').minute)%15)
    s.drop(['TIMESTAMP'], axis=1, inplace=True)

    s.drop(['POLYLINE'], axis=1, inplace=True)
    s.drop(['MISSING_DATA'], axis=1, inplace=True)
    s.drop(['x'], axis=1, inplace=True)

    s['ORIGIN_CALL'] = s['ORIGIN_CALL'].fillna(0)
    s['ORIGIN_STAND'] = s['ORIGIN_STAND'].fillna(0)

    s['ORIGIN_CALL'] = s['ORIGIN_CALL'].apply(lambda x : category_mapping_origin_call.get(x, 0))
    s['ORIGIN_STAND'] = s['ORIGIN_STAND'].apply(lambda x : category_mapping_origin_stand.get(x, 0))
    s['TAXI_ID'] = s['TAXI_ID'].apply(lambda x : category_mapping_taxi_ID.get(x, 0))

    s['CALL_TYPE_A'] = s['CALL_TYPE'].apply(lambda x : 1 if x == 'A' else 0)
    s['CALL_TYPE_B'] = s['CALL_TYPE'].apply(lambda x : 1 if x == 'B' else 0)
    s['CALL_TYPE_C'] = s['CALL_TYPE'].apply(lambda x : 1 if x == 'C' else 0)

    s.drop(['CALL_TYPE'], axis=1, inplace=True)
    s.drop(['DAY_TYPE'], axis=1, inplace=True)

    if test_set:
        return s

    size = temp_size

    if len(s) >= size:
        return s
    else: 
        s = pd.concat([s, process_set(dataset, size - len(s))])
        return s

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def normalize_X(X, test = False):
    global scaler
    columns_to_scale = [col for col in X.columns if col not in catego_features and col not in ['y1', 'y2']]
    if not test:
        X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    else:
        X[columns_to_scale] = scaler.transform(X[columns_to_scale])

    return X


category_mapping_origin_call = {}
category_mapping_origin_stand = {}
category_mapping_taxi_ID = {}

catego_features = ['CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'ORIGIN_STAND', 'TAXI_ID', 'month', 'day']

lower_bound_len, upper_bound_len = -1, 185
lower_bound_y1, upper_bound_y1 = -8.789,-8.284
lower_bound_y2, upper_bound_y2 = 40.892,41.397

data = None
data_test = None

SIZE = None
f = 0.3
k = 4

def init():
    print("- INIT METHOD ", flush = True)
    global category_mapping_origin_call
    global category_mapping_origin_stand
    global category_mapping_taxi_ID


    global data
    global data_test
    print("-- csv loading ", flush = True)
    print("--- csv loading : train", flush = True)
    file_path = '../data/train.csv'
    data = pd.read_csv(file_path, index_col="TRIP_ID")

    data_LS = data.sample(frac=1-f, replace=False, random_state=SEED)
    data_VS = data.drop(data_LS.index) #In this code VS is equivalent to TS_artificial of the report. We use VS since we don't want to confuse it with the TS of the competition.

    global SIZE
    SIZE = len(data)

    print("--- csv loading : test", flush = True)
    file_path_test = '../data/test.csv'
    data_test = pd.read_csv(file_path_test, index_col="TRIP_ID")

    category_frequencies_call = data['ORIGIN_CALL'].value_counts(normalize=True)
    category_frequencies_stand = data['ORIGIN_STAND'].value_counts(normalize=True)
    category_frequencies_taxi = data['TAXI_ID'].value_counts(normalize=True)

    top_categories_call = category_frequencies_call.nlargest(254).index
    top_categories_stand = category_frequencies_stand.nlargest(254).index
    top_categories_taxi = category_frequencies_taxi.nlargest(254).index

    category_mapping_origin_call = {category: i for i, category in enumerate(top_categories_call)}
    category_mapping_origin_stand = {category: i for i, category in enumerate(top_categories_stand)}
    category_mapping_taxi_ID = {category: i for i, category in enumerate(top_categories_taxi)}

    print(f"-- Processing the LS and the VS (total : {SIZE}) ~ {(1-f)}-{f}", flush = True)
    print(f"--- Processing the LS (total : {SIZE*(1-f)} ~ {(1-f)})", flush = True)
    LS = process_set(data_LS, SIZE*(1-f), test_set = False, validation=False)
    print(f"--- Processing the VS (total : {SIZE*(f)})~ {(f)})", flush = True)
    VS = process_set(data_VS, SIZE*f, test_set = False, validation=True)

    X_train, y_train = LS.drop(['y1','y2'], axis=1), LS[['y1', 'y2']]
    X_val, y_val = VS.drop(['y1','y2'], axis=1), VS[['y1', 'y2']]
    print("-- Processing the TS", flush = True)
    TS = process_set(data_test, None, test_set = True)
    TS = TS.drop(['y1', 'y2'], axis=1)

    return [X_train, y_train], [X_val, y_val], [TS, None]


def write_submission(trip_ids, destinations, file_name="submission"):
    """
    This function writes a submission csv file given the trip ids, 
    and the predicted destinations.

    Parameters
    ----------
    trip_id : List of Strings
        List of trip ids (e.g., "T1").
    destinations : NumPy Array of Shape (n_samples, 2) with float values
        Array of destinations (latitude and longitude) for each trip.
    file_name : String
        Name of the submission file to be saved.
        Default: "submission".
    """
    n_samples = len(trip_ids)
    assert destinations.shape == (n_samples, 2)

    submission = pd.DataFrame(
        data={
            'LATITUDE': destinations[:, 1],
            'LONGITUDE': destinations[:, 0],
        },
        columns=["LATITUDE", "LONGITUDE"],
        index=trip_ids,
    )

    # Write file
    submission.to_csv(file_name + ".csv", index_label="TRIP_ID")