# Let's get into Deep learning using TensorFlow
All the contributions, licenses and references are mentioned in the LICENSE file.

## How to install TensorFlow:
Install tensorflow on your machine.  
run the command **pip install tensorflow** on the command prompt.  
* If you get an error specifying 'There is no module named tensorflow for the current version of python'.  
  * Just downgrade your python from 3.8.x to 3.7.x or better 3.6.x  
  * Do not forget to check the add Python to path checkbox while installing the other version of python.
* If the module is installed successfully but the code does not run.  
  * Run **pip freeze** on the command prompt. It lists all the python modules installed on your machine.
  * Find the version of tensorflow installed on the machine.
  * If it is 2.1x then downgrade it to 2.0 by running the following commands:  
  **pip uninstall tensorflow** (to uninstall the current version)  
  **pip install tensorflow==2.0** (to install the 2.0 version)  

visit [TensorFlow](https://www.tensorflow.org/api_docs/python/tf) for more information.  

## Get your code:
Import the necessary modules for training the model.  
There are not many imports you need, if you use tensorflow.  
Imports refer to the modules required to perform certain tasks or numerical operations.  
Refer **TensorFlow_MNIST.py**, where you will know how to train a deep neural network.  
It's pretty straight-forward and very readable.

## The Progressbar issue:  
You can run the code **TensorFlow_MNIST.py** on your code editor.  
But it's better to use ipython notebooks or command prompt because some of the progress bars which are used to refer the status of the training process fail to run properly on some code editors.  

## References:  
If you want to know how neural networks work or you are curious about the theoretical part.  
Here are some sources.  
* [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [MIT](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
