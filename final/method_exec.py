import method
import numpy as np
np.random.seed(method.SEED)

from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


(X_train, Y_train), (X_val, Y_val), (X_test, _) = method.init()

# Hyperparameter grid for RidgeCV (final estimator)
ridge_param_grid = {
    'alphas': [0.01, 0.05, 0.1, 0.5, 1.0]
}

XGB_model_long = XGBRegressor(n_jobs=6, verbosity=1, n_estimators=1000, max_depth=10, eta=0.01, subsample=0.7, colsample_bytree=0.7, random_state=method.SEED)
RF_model_long = RandomForestRegressor(n_jobs=6, n_estimators=200, max_depth=20, min_samples_leaf=5, random_state=method.SEED)
HGB_model_long = HistGradientBoostingRegressor(categorical_features=method.catego_features, learning_rate=0.1, max_iter=1000, max_depth=8, min_samples_leaf=5, max_leaf_nodes=100, random_state=method.SEED)
#Cat_model_long = CatBoostRegressor(cat_features=method.catego_features, iterations=1000, learning_rate=0.1, depth=6, verbose=1)
#MLP_model_long = MLPRegressor(hidden_layer_sizes=(200, 200, 100, 50), verbose=1)

XGB_model_lat = XGBRegressor(n_jobs=6, verbosity=1, n_estimators=1000, max_depth=10, eta=0.01, subsample=0.7, colsample_bytree=0.7, random_state=method.SEED+1)
RF_model_lat= RandomForestRegressor(n_jobs=6, n_estimators=200, max_depth=40, min_samples_leaf=5, random_state=method.SEED+1)
HGB_model_lat = HistGradientBoostingRegressor(categorical_features=method.catego_features, learning_rate=0.01, max_iter=1000, max_depth=10, min_samples_leaf=5, max_leaf_nodes=200,random_state=method.SEED+1)
#Cat_model_lat = CatBoostRegressor(cat_features=method.catego_features, iterations=1000, learning_rate=0.1, depth=6, verbose=1)
#MLP_model_lat = MLPRegressor(hidden_layer_sizes=(200, 200, 100, 50), verbose=1)

estimators_long =    [
                    ('XGB', XGB_model_long),
                    ('RF', RF_model_long),
                    ('HGB', HGB_model_long),
                    #('Cat', Cat_model_long),
                    #('MLP', MLP_model_long)
                ]
estimators_lat =    [
                    ('XGB', XGB_model_lat),
                    ('RF', RF_model_lat),
                    ('HGB', HGB_model_lat),
                    #('Cat', Cat_model_lat),
                    #('MLP', MLP_model_lat)
                ]

final_long = RidgeCV(alphas=ridge_param_grid['alphas'])
final_lat = RidgeCV(alphas=ridge_param_grid['alphas'])

print("- MODEL FITTING : long", flush=True)

stacker_long = StackingRegressor(estimators=estimators_long,final_estimator=final_long, n_jobs=6, verbose=1)
stacker_long.fit(X_train, Y_train['y1'])

print("- MODEL FITTING : lat", flush=True)

stacker_lat = StackingRegressor(estimators=estimators_lat,final_estimator=final_lat, n_jobs=6, verbose=1)
stacker_lat.fit(X_train, Y_train['y2'])

print("- MODEL PREDICTING : long & lat", flush=True)

y_pred_train_long, y_pred_train_lat = stacker_long.predict(X_train), stacker_lat.predict(X_train)
y_pred_val_long, y_pred_val_lat = stacker_long.predict(X_val), stacker_lat.predict(X_val)
y_pred_test_long, y_pred_test_lat = stacker_long.predict(X_test), stacker_lat.predict(X_test)

y_pred_train = np.column_stack((y_pred_train_long, y_pred_train_lat))
y_pred_val = np.column_stack((y_pred_val_long, y_pred_val_lat))
y_pred_test = np.column_stack((y_pred_test_long, y_pred_test_lat))

score_train = np.mean(method.haversine(y_pred_train, Y_train.values))
score_val = np.mean(method.haversine(y_pred_val, Y_val.values))

print(f"SCORE : train = {score_train}, TS_artificial = {score_val}", flush=True)

method.write_submission(X_test.index, y_pred_test, file_name='tmp')
print(f"-- WRITTED -- ", flush=True)
