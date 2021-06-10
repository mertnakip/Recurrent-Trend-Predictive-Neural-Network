# Recurrent Trend Predictive Network with Application to Multi-Sensor Fire Detection

This repository contains the codes of the Keras implementation of the Recurrent Trend Predictive Neural Network (rTPNN) model, as well as an example for the multi-sensor fire detection in the folder CrossValidationOnFireDataset.

Note that it is an particular implementation of rTPNN model and it may be implemented in different ways.


## Usage of rTPNN 

from rTPNN import rTPNN

r_tpnn = rTPNN(num_features, predictor_arch, activation_name='sigmoid')  
\# num_signals: total number of time series features  
\# predictor_arch: a list of number of neurons for fully connected layers (len(predictor_arch) = number of layers; predictor_arch[0] = number of neurons at the first fully connected layer)

rTPNN_model = r_tpnn.model  
rTPNN_model.fit(x_train, y_train, epochs=num_epochs, batch_size=size_of_a_batch) # equivalent to the "fit" method of a Keras model  
prediction = rTPNN_model.predict(x_test)

\# Note that rTPNN model contains all methods and attributes that have been contained by any Keras model.

\# x_train is a list of time series data (len(x_train) = num_features; x_train[0]: numpy array; x_train[0].shape = (num_train_samples, num_past=2, 1))  
\# Similarly, x_test is a list of time series data (len(x_test) = num_features; x_test[0]: numpy array; x_test[0].shape = (num_test_samples, num_past=2, 1)) 
 

## Citation Request
The rTPNN as well as its application on multi-sensor fire detection has been published as a journal paper which is entitled as "Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection" on IEEE Access. If you use rTPNN architectur or the content of this repository, please cite the following our paper as follows: 


