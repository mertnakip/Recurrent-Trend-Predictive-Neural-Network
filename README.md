### This repository will be updated to be improved further..

# Recurrent Trend Predictive Network with Application to Multi-Sensor Fire Detection

This repository contains the codes of the Keras implementation of the Recurrent Trend Predictive Neural Network (rTPNN) model, as well as an example for the multi-sensor fire detection in the folder CrossValidationOnFireDataset.

You may find the more detailed explanation of the methodology as well as the results in our publication at https://ieeexplore.ieee.org/document/9451553.

Note that it is an particular implementation of rTPNN model and it may be implemented in different ways.

## (Classic) Numpy Array Input To List which is Desired by (this implementation of) rTPNN

from ConvertDataForRTPNN import convert_data_for_rTPNN

x_list = convert_data_for_rTPNN(x)

Provide array "x" to "convert_data_for_rTPNN" as shown in the following figure. 

![alt text](https://www.researchgate.net/publication/352306006/figure/fig3/AS:1033351641198595@1623381638599/The-dimensions-of-the-input-tensor-and-output-vector-of-rTPNN-in-the-case-where-all_W640.jpg)

## Usage of rTPNN 

from rTPNN import rTPNN

r_tpnn = rTPNN(num_features, predictor_arch, activation_name='sigmoid')  
\# num_signals: total number of time series features  
\# predictor_arch: a list of number of neurons for fully connected layers (len(predictor_arch) = number of layers; predictor_arch[0] = number of neurons at the first fully connected layer)

rTPNN_model = r_tpnn.model  
rTPNN_model.compile(optimizer='adam', loss='mse')
rTPNN_model.fit(x_train, y_train, epochs=num_epochs, batch_size=size_of_a_batch) # equivalent to the "fit" method of a Keras model  
prediction = rTPNN_model.predict(x_test)

\# Note that rTPNN model contains all methods and attributes that have been contained by any Keras model.

\# x_train is a list of time series data (len(x_train) = num_features; x_train[0]: numpy array; x_train[0].shape = (num_train_samples, num_past=2, 1))  
\# Similarly, x_test is a list of time series data (len(x_test) = num_features; x_test[0]: numpy array; x_test[0].shape = (num_test_samples, num_past=2, 1)) 
 

## Citation Request
The rTPNN as well as its application on multi-sensor fire detection has been published as a journal paper which is entitled as "Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection" on IEEE Access. If you use rTPNN architectur or the content of this repository, please cite the following our paper as follows: 

@ARTICLE{9451553,
  author={Nakip, Mert and Güzelíş, Cüneyt and Yildiz, Osman},
  journal={IEEE Access}, 
  title={Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2021.3087736}
  }
