import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

x_data = np.load("NN_playerTrain_x_data.npy")
y_data = np.load("NN_playerTrain_y_data.npy")

dct = DecisionTreeClassifier()
dct.fit(x_data, y_data)

# Save the trained model
joblib.dump(dct, "trained_model.pkl")
