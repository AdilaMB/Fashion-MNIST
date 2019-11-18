import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trainig = pd.read_csv('E:/Maestrado/Machine Learning/Machine Learning I/DataSet/MNIST/fashion-mnist_train.csv',
                              sep=';', header=None)
test = pd.read_csv('E:/Maestrado/Machine Learning/Machine Learning I/DataSet/MNIST/fashion-mnist_test.csv',
                       sep=';', header=None)

    # Divido los datos del trining y test
X_trainig = trainig.iloc[1:, 1:].values
Y_trainig = trainig.iloc[1:, 0:1].values.reshape(-1, 1)
Y_trainig = Y_trainig.astype('int')
X_trainig = X_trainig.astype('int')

X_test = test.iloc[1:, 1:].values
Y_test = test.iloc[1:, 0:1].values.reshape(-1, 1)
X_test = X_test.astype('int')
Y_test = Y_test.astype('int')

learning_rate = 0.050

model = keras.Sequential()
model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dense(256, activation=tf.nn.softmax))

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# This builds the model for the first time:
model = model.fit(X_trainig, Y_trainig, batch_size=1000, epochs=5)

#acc= tf.keras.metrics.categorical_accuracy(Y_test, y_pred)

#print(acc)

