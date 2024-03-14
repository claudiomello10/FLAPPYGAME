import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.neighbors import KNeighborsClassifier
import joblib


x_data = np.load("NN_playerTrain_x_data.npy")
y_data = np.load("NN_playerTrain_y_data.npy")


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_data, y_data)


joblib.dump(knn, "knn_model.pkl")
