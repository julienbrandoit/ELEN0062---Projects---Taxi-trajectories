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
#from catboost import CatBoostRegressor
#from sklearn.neural_network import MLPRegressor

(X_train, Y_train), (X_val, Y_val), (X_test, _) = method.init()


RF_model_long = RandomForestRegressor(n_jobs=-1, n_estimators=200, max_depth=40, min_samples_leaf=5)

RF_model_lat= RandomForestRegressor(n_jobs=-1, n_estimators=200, max_depth=40, min_samples_leaf=5)

print("- MODEL FITTING : long", flush=True)

RF_model_long.fit(X_train, Y_train['y1'])

print("- MODEL FITTING : lat", flush=True)

RF_model_lat.fit(X_train, Y_train['y2'])

print("- MODEL PREDICTING : long & lat", flush=True)

feature_importances_long = RF_model_long.feature_importances_
feature_importances_lat = RF_model_lat.feature_importances_

import matplotlib.pyplot as plt
# Noms des fonctionnalités (replace with your actual feature names)
feature_names = X_train.columns

# Création d'un tableau pour l'axe y
y_axis = np.arange(len(feature_names))

# Largeur des barres
bar_width = 0.4

# Création du graphique
plt.figure(figsize=(12, 8))

# Barres pour long
plt.barh(y_axis - bar_width/2, feature_importances_long, height=bar_width, align='center', label='Longitude', color='blue', alpha=0.7)

# Barres pour lat
plt.barh(y_axis + bar_width/2, feature_importances_lat, height=bar_width, align='center', label='Latitude', color='green', alpha=0.7)

plt.yticks(y_axis, feature_names)
plt.xlabel('Importance')
plt.title('Feature Importances - Longitude and Latitude')
plt.legend()

plt.show()

print(f"FEATURE IMPORTANCES : long = {feature_importances_long}, lat = {feature_importances_lat}", flush=True)

y_pred_train_long, y_pred_train_lat = RF_model_long.predict(X_train), RF_model_lat.predict(X_train)
y_pred_val_long, y_pred_val_lat = RF_model_long.predict(X_val), RF_model_lat.predict(X_val)
y_pred_test_long, y_pred_test_lat = RF_model_long.predict(X_test), RF_model_lat.predict(X_test)

y_pred_train = np.column_stack((y_pred_train_long, y_pred_train_lat))
y_pred_val = np.column_stack((y_pred_val_long, y_pred_val_lat))
y_pred_test = np.column_stack((y_pred_test_long, y_pred_test_lat))

score_train = np.mean(method.haversine(y_pred_train, Y_train.values))
score_val = np.mean(method.haversine(y_pred_val, Y_val.values))

print(f"SCORE : train = {score_train}, validation = {score_val}", flush=True)

method.write_submission(X_test.index, y_pred_test, file_name='tmp')
print(f"-- WRITTED -- ", flush=True)
