# Vanila-Python-Neural-Network
Neural network building application only supporting fully connected layers. 

The project is for educational purposes to see how the back-propagation works. 

To chose activation function call the model with either 'relu' or 'sigmoid'.

The app is tested on MNIST dataset but technically should work on other datasets aswell.  Change the load script accordingly. 

Known issues:

The weight saving function is incomplete does not work properly. 

The algorithm only works on cpu which means it's extremely slow as you increase the number of layers. 

Stochastic gradient decent does not work set the number to full size of training set so it acts as batch gradient decent. 

