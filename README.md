# Neural-Network
 3 layered deep neural network for training 5000 datasets from the famous MNIST database.

This architecture has only 1 hidden layer. 
The hidden layer consists of 25 neurons(nodes) and the input layer consists of 400 different features,i.e.pixels.
I have randomly initialised the Weight matrix (Theta1 and Theta2) of shape (25,401) and (10,26)
Note:-The column sizes of two weight matrices are 401 and 26 because of bias node that has to be added.

Packages used - numpy,scipy.sio,pandas
Accuracy achieved by this model for learning_rate=2.00 and no.of itertions = 600 : 95.54%

Activation function-sigmoid.
