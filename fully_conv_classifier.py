import time
b = time.time()
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import os 
from pathlib import Path
a = time.time()
print('Imports complete in ', a-b, ' seconds')

def plot_loss(val_loss_list):
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    xs = [i+1 for i in range(len(val_loss_list))]
    plt.plot(xs, val_loss_list, color='b', linewidth=0.8, label = 'val_loss')
    plt.axvline(x=val_loss_list.index(min(val_loss_list)) +
                1, color='g', linewidth=0.7, label = 'min_loss')
    plt.axhline(y = min(val_loss_list), color = 'g', linewidth = 0.7)
    plt.legend()
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

model = models.Sequential()
model.add(layers.Conv2D(300, (3, 3), (2, 2),
                        input_shape=(28, 28, 1), activation='relu'))
model.add(layers.Conv2D(100, (3, 3), (2, 2), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(10, (5, 5), (2, 2), activation='softmax'))
model.add(layers.Reshape((10,)))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('Model compiled.')

(X_train, y_train),(X_test, y_test) = datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


#_________TWEAKABLES____________
PATIENCE = 4
BATCH_SIZE = 30
VERBOSE = False
PLT_SHOW = 3
SAVE_MODEL = False
checkpoint_filename = 'mnist'
# ______________________________


val_loss_list = []
val_acc_list = []
epoch_count = 1
train_flag = True
patience_temp = PATIENCE

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) != 0:
    device = tf.device('GPU:0')
    print('Running on the GPU.')
else:
    device = tf.device('CPU')
    print('Running on the CPU.')

with device:
    while train_flag:
        model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                  epochs=1, verbose=VERBOSE)
        val_loss, val_acc = model.evaluate(X_test, y_test, verbose=VERBOSE)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc*100)
        print(f'Epoch {epoch_count}, Acc {round(val_acc*100, 3)}, Loss {round(val_loss, 5)}.')
        # train_optimizer part
        if val_loss != min(val_loss_list):
            patience_temp -= 1
        else:
            patience_temp = PATIENCE
            if SAVE_MODEL is True:
                if Path(f'./{checkpoint_filename}').is_file():
                    os.remove(f'./{checkpoint_filename}')
                model.save(f'./{checkpoint_filename}')
                if VERBOSE is True:
                    print(f'New model saved as <{checkpoint_filename}>.')
            else:
                pass
            
        if patience_temp == 0:
            train_flag = False
        else:
            train_flag = True
        #Plot metrics
        if epoch_count % PLT_SHOW == 0:
            plot_loss(val_loss_list)
        epoch_count += 1
plot_loss(val_loss_list)