import method
import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

(X_train, Y_train), (X_val, Y_val), (X_test, _) = method.init()

# Hyperparameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [5, 7, 10],
    'eta': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [20, 30, 40],
    'min_samples_leaf': [5, 10, 15]
}

# Hyperparameter grid for HistGradientBoosting
hgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [1000, 1500, 2000],
    'max_depth': [8, 10, 12],
    'min_samples_leaf': [5, 10, 15],
    'max_leaf_nodes': [100, 200, 300]
}

# Hyperparameter grid for RidgeCV (final estimator)
ridge_param_grid = {
    'alphas': [0.01, 0.05, 0.1, 0.5, 1.0]
}

print("- HYPERPARAMETER TUNING : long", flush=True)

# Create GridSearchCV objects for each base model
xgb_grid_search = GridSearchCV(XGBRegressor(n_jobs=7, verbosity=1), xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=7, verbose=1)
rf_grid_search = GridSearchCV(RandomForestRegressor(n_jobs=7, verbose=1), rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=7, verbose=1)
hgb_grid_search = GridSearchCV(HistGradientBoostingRegressor(verbose=1, categorical_features=method.catego_features), hgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=7, verbose=1)

# Fit the GridSearchCV objects
print("-- HYPERPARAMETER rf_grid_search", flush=True)
#rf_grid_search.fit(X_train, Y_train['y1'])
#best_rf_params_long = rf_grid_search.best_params_
#print("--- rf :", best_rf_params_long, flush=True)
#rf_grid_search = GridSearchCV(RandomForestRegressor(n_jobs=7, verbose=1), rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=7, verbose=1)
#rf_grid_search.fit(X_train, Y_train['y2'])
#best_rf_params_lat = rf_grid_search.best_params_
#print("--- rf :", best_rf_params_lat, flush=True)
print("-- HYPERPARAMETER hgb_grid_search", flush=True)
hgb_grid_search.fit(X_train, Y_train['y1'])
best_hgb_params_long = hgb_grid_search.best_params_
print("-- hgb:", best_hgb_params_long, flush=True)
exit()
hgb_grid_search = GridSearchCV(HistGradientBoostingRegressor(verbose=1, categorical_features=method.catego_features), hgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=7, verbose=1)
hgb_grid_search.fit(X_train, Y_train['y2'])
best_hgb_params_lat = hgb_grid_search.best_params_
print("-- hgb:", best_hgb_params_lat, flush=True)
print("-- HYPERPARAMETER xgb_grid_search", flush=True)
xgb_grid_search.fit(X_train, Y_train['y1'])
best_xgb_params_long = xgb_grid_search.best_params_
print("-- xgb : ",best_xgb_params_long, flush=True)
xgb_grid_search = GridSearchCV(XGBRegressor(n_jobs=7, verbosity=1), xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=7, verbose=1)
xgb_grid_search.fit(X_train, Y_train['y2'])
best_xgb_params_lat = xgb_grid_search.best_params_
print("-- xgb : ",best_xgb_params_lat, flush=True)

# Get the best hyperparam
exit()
print("-- BEST params:", flush=True)


print("- HYPERPARAMETER TUNING : lat", flush=True)

# Create GridSearchCV objects for each base model

# Fit the GridSearchCV objects
print("-- HYPERPARAMETER rf_grid_search", flush=True)
print("-- HYPERPARAMETER hgb_grid_search", flush=True)
print("-- HYPERPARAMETER xgb_grid_search", flush=True)

# Get the best hyperparameters from the grid search

print("-- BEST params:", flush=True)
print("-- xgb : ",best_xgb_params_lat, flush=True)
print("-- rf :", best_rf_params_lat, flush=True)
print("-- hgb:", best_hgb_params_lat, flush=True)

exit()

XGB_model_long = XGBRegressor(n_jobs=7)
RF_model_long = RandomForestRegressor(n_jobs=7)
HGB_model_long = HistGradientBoostingRegressor(categorical_features=method.catego_features)

XGB_model_lat = XGBRegressor(n_jobs=7)
RF_model_lat = RandomForestRegressor(n_jobs=7)
HGB_model_lat = HistGradientBoostingRegressor(categorical_features=method.catego_features)


# Update the base models with the best hyperparameters
XGB_model_long.set_params(**best_xgb_params_long)
RF_model_long.set_params(**best_rf_params_long)
HGB_model_long.set_params(**best_hgb_params_long)

XGB_model_lat.set_params(**best_xgb_params_lat)
RF_model_lat.set_params(**best_rf_params_lat)
HGB_model_lat.set_params(**best_hgb_params_lat)

estimators_long =    [
                    ('XGB', XGB_model_long),
                    ('RF', RF_model_long),
                    ('HGB', HGB_model_long)
                ]
estimators_lat =    [
                    ('XGB', XGB_model_lat),
                    ('RF', RF_model_lat),
                    ('HGB', HGB_model_lat)
                ]

final_long = RidgeCV(alphas=ridge_param_grid['alphas'])
final_lat = RidgeCV(alphas=ridge_param_grid['alphas'])

print("- MODEL FITTING : long", flush=True)

stacker_long = StackingRegressor(estimators=estimators_long,final_estimator=final_long, n_jobs=7, verbose=1)
stacker_long.fit(X_train, Y_train['y1'])

print("- MODEL FITTING : lat", flush=True)

stacker_lat = StackingRegressor(estimators=estimators_lat,final_estimator=final_lat, n_jobs=7, verbose=1)
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

print(f"SCORE : train = {score_train}, validation = {score_val}", flush=True)

method.write_submission(X_test.index, y_pred_test, file_name='stacker_10.csv')