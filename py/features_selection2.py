import method
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import numpy as np

(X_train, Y_train), (X_val, Y_val), (X_test, _) = method.init()
from sklearn.feature_selection import mutual_info_regression

# Calculate mutual information for longitude (y1)
feature_importances_mi_long = mutual_info_regression(X_train, Y_train['y1'])

# Calculate mutual information for latitude (y2)
feature_importances_mi_lat = mutual_info_regression(X_train, Y_train['y2'])

# Noms des fonctionnalités (replace with your actual feature names)
feature_names = X_train.columns

# Création d'un tableau pour l'axe y
y_axis = np.arange(len(feature_names))

# Largeur des barres
bar_width = 0.4
# Création du graphique avec une taille de texte plus grande
plt.figure(figsize=(12, 8))

# Barres pour long (mutual information)
plt.barh(y_axis - bar_width/2, feature_importances_mi_long, height=bar_width, align='center', label='Longitude', color='blue', alpha=0.7)

# Barres pour lat (mutual information)
plt.barh(y_axis + bar_width/2, feature_importances_mi_lat, height=bar_width, align='center', label='Latitude', color='green', alpha=0.7)

plt.yticks(y_axis, feature_names, fontsize=14)  # Ajuster la taille du texte sur l'axe y
plt.xlabel('Mutual Information', fontsize=16)  # Ajuster la taille du texte de l'axe x
# Ajuster la taille des valeurs sur l'axe x
plt.tick_params(axis='x', labelsize=14)
#plt.title('Mutual Information - Longitude and Latitude', fontsize=18)  # Ajuster la taille du titre
plt.legend(fontsize=14)  # Ajuster la taille de la légende

plt.show()