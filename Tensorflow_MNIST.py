#The very basic imports for a feedforward deep_learning model in tensorflow.

import time
bef = time.time()
import tensorflow as tf 
from tensorflow.keras import datasets, models, layers
aft = time.time()
print(f'Imports complete in {aft-bef} seconds.')


#Load MNIST dataset (consisits of labeled hand-written digits from 0 t0 9. Each image is 28x28 pixels in size).
#split the data into training and testing data.

(images_train, labels_train), (images_test, labels_test) = datasets.mnist.load_data() 
print('Data processing complete.')



'''
Creating the model is very easy in tensorflow.
You can add as many layers as you wish provided the input shape and the 
number of output units remains the same for a specific problem.
'''


model = models.Sequential([

    layers.Flatten(input_shape = (28,28)),      #The flatten layer to flatten the 2D(28x28 px) image into a 1D(28*28=784) array.
    layers.Dense(128, activation = 'elu'),      #Hidden layer 1 with elu activation for non linearity.(Dense = fully connected layer) 
    layers.Dense(128, activation = 'elu'),      #Hidden layer 2 with elu activation for non linearity.
    layers.Dense(32, activation = 'elu'),       #Hidden layer 3 with elu activation for non linearity.
    layers.Dense(10, activation = 'softmax')    #Output layer with softmax activation.

])




#Mention the loss function and the optimizer you want to use before fitting the model.
#You can use many other loss functions and optimizer Visit tensorflow.org docs for more information.

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



'''
Now the model is ready for training.
If you want to train the model 
by specifying the number of epochs for which the model is to be trained
you can use the syntax below.
It gives you the train_loss and test_loss(Val_loss) for each epoch.
The argument 'validation_data' is optional.
Method 1:
'''

model.fit(images_train, labels_train, epochs = 6, validation_data = (images_test, labels_test))
#the model is now trained for 6 epochs




'''
Or else, if you want to monitor data and set up some logic 
so as to stop training when the model reaches certain accuracy.
You can use the code below.
Method 2:
'''

val_acc = 0
while val_acc < 0.97:      #Target accuracy: 97 percent
    model.fit(images_train, labels_train, epochs = 1)
    val_loss, val_acc = model.evaluate(images_test, labels_test)
#This way the model stops training when the validation accuracy reaches 97 percent.






