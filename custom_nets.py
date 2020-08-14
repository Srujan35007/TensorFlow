import time 
b = time.time()
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
from tensorflow.keras import models, layers, Sequential
import numpy as np
import matplotlib.pyplot as plt 
a = time.time()
print(f'Imports complete in {a-b} seconds')

