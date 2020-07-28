import time 
b = time.time()
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras import layers, models, datasets
a = time.time()
print(f'Imports complete in {a-b} seconds')

def get_flat_shape(foo_shape):
    prod = 1
    for i in range(len(foo_shape)):
        prod = prod * foo_shape[i]
    return prod

# __________TWEAKABLES____________
raw_input_shape = (28,28,1) # Height Width Channels
flat_input_shape = get_flat_shape(raw_input_shape)
hidden_layer_units = [200,100,30]
n_out_classes = 10
classification = True
# ________________________________


model = models.Sequential()
model.add(layers.Flatten(input_shape = raw_input_shape))
model.add(layers.Dense(flat_input_shape, activation='relu'))
for i in range(len(hidden_layer_units)):
    model.add(layers.Dense(hidden_layer_units[i], activation='relu'))
if classification:
    model.add(layers.Dense(n_out_classes, activation='softmax'))
else:
    model.add(layers.Dense(n_out_classes, activation='sigmoid'))


model.summary()
loss_fn = 'sparse_categorical_crossentropy' if classification else 'mse'
model.compile(loss = loss_fn, optimizer='adam', metrics=['acciracy'])
print('Model compiled')