import time 
b = time.time()
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras import datasets, models, layers, Sequential
import numpy as np
import matplotlib.pyplot as plt 
a = time.time()
print(f'Imports complete in {a-b} seconds')

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = (X_train/255).reshape(60000,28*28)
X_test = (X_test/255).reshape(10000, 28*28)

encoder = Sequential([
    layers.Dense(100, activation='relu', input_shape = (28*28,)),
    layers.Dense(50, activation='relu'),
    layers.Dense(10, activation='linear')
])

decoder = Sequential([
    layers.Dense(50, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(28*28, activation='relu')
])

model = Sequential([encoder, decoder])
enc = encoder
dec = decoder

model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])
model.fit(X_train, X_train, epochs = 1)