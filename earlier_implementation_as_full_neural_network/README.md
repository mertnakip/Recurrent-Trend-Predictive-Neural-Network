# Recurrent Trend Predictive Neural Network with Application to Multi-Sensor Fire Detection

![alt text](https://www.researchgate.net/publication/352306006/figure/fig1/AS:1033351641178112@1623381638542/The-architecture-of-the-Recurrent-Trend-Predictive-Neural-Network-rTPNN_W640.jpg)

This repository contains the codes of the Keras implementation of the Recurrent Trend Predictive Neural Network (rTPNN) model, as well as an example for the multi-sensor fire detection in the folder CrossValidationOnFireDataset.

You may find the more detailed explanation of the methodology as well as the results in our publication at https://ieeexplore.ieee.org/document/9451553.

Note that it is an particular implementation of rTPNN model and it may be implemented in different ways.

## (Classic) Numpy Array Input To List which is Desired by (this implementation version of) rTPNN

from ConvertDataForRTPNN import convert_data_for_rTPNN

x_list = convert_data_for_rTPNN(x)

Provide array "x" to "convert_data_for_rTPNN" as shown in the following figure. 

![alt text](https://www.researchgate.net/publication/352306006/figure/fig3/AS:1033351641198595@1623381638599/The-dimensions-of-the-input-tensor-and-output-vector-of-rTPNN-in-the-case-where-all_W640.jpg)

## Usage of rTPNN 

from rTPNN import rTPNN

r_tpnn = rTPNN(num_features, predictor_arch, activation_name='sigmoid')  
\# num_features: total number of time series features  
\# predictor_arch: a list of number of neurons for fully connected layers (len(predictor_arch) = number of layers; predictor_arch[0] = number of neurons at the first fully connected layer)

rTPNN_model = r_tpnn.model  
rTPNN_model.compile(optimizer='adam', loss='mse')  
rTPNN_model.fit(x_train, y_train, epochs=num_epochs, batch_size=size_of_a_batch) # equivalent to the "fit" method of a Keras model  
prediction = rTPNN_model.predict(x_test) 

\# Note that rTPNN model contains all methods and attributes that have been contained by any Keras model. 

\# x_train is a list of time series data (len(x_train) = num_features; x_train[0]: numpy array; x_train[0].shape = (num_train_samples, num_past=2, 1))  
\# Similarly, x_test is a list of time series data (len(x_test) = num_features; x_test[0]: numpy array; x_test[0].shape = (num_test_samples, num_past=2, 1)) 
 
 
## An example of rTPNN on Random Data

import numpy as np  
from ConvertDataForRTPNN import convert_data_for_rTPNN  
from rTPNN import rTPNN  

T = 100; 
I = 5

x = np.random.rand(T, 2, I)  
y = np.random.rand(T) 

x_list = convert_data_for_rTPNN(x) 

r_tpnn = rTPNN(I, [I*2, I, np.ceil(I/2)], activation_name='sigmoid')  
rTPNN_model = r_tpnn.model  
rTPNN_model.compile(optimizer='adam', loss='mse')  
rTPNN_model.fit(x_list, y, epochs=10, batch_size=20, verbose=0)  
prediction = rTPNN_model.predict(x_list) 

## Citation Request 
The rTPNN as well as its application on multi-sensor fire detection has been published as a journal paper which is entitled as "Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection" on IEEE Access. If you use rTPNN architecture or the content of this repository, please cite our following paper as follows: 

@ARTICLE{nakip2021rTPNN,  
  author={Nakip, Mert and Güzelíş, Cüneyt and Yildiz, Osman},  
  journal={IEEE Access},  
  title={Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection},  
  year={2021},  
  volume={9},  
  number={},  
  pages={84204-84216},  
  doi={10.1109/ACCESS.2021.3087736}  
  }
