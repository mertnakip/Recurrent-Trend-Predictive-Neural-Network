# Recurrent Trend Predictive Neural Network

![alt text](https://www.researchgate.net/publication/352306006/figure/fig1/AS:1033351641178112@1623381638542/The-architecture-of-the-Recurrent-Trend-Predictive-Neural-Network-rTPNN_W640.jpg)

This repository contains the implementation of the Recurrent Trend Predictive Neural Network (rTPNN) model as a Keras layer. In addition, it also contains an application of rTPNN for the multi-sensor fire detection in the folder FireDetection_via_rTPNN.

You may find the more detailed explanation of the methodology as well as the results in our publication at https://ieeexplore.ieee.org/document/9451553.

Note that it is an particular implementation of rTPNN, and it may be implemented in different ways.

## Inputs for rTPNN Layer

Provide input array "x" as shown in the following figure. 

![alt text](https://www.researchgate.net/publication/352306006/figure/fig3/AS:1033351641198595@1623381638599/The-dimensions-of-the-input-tensor-and-output-vector-of-rTPNN-in-the-case-where-all_W640.jpg)

## An example usage of rTPNN 

import numpy as np  
from keras.layers import Input, Dense
from keras import Model
from rTPNN_layer import rTPNN  


###### Random Data

num_samples = 100; 
num_features = 5

x = np.random.rand(num_samples, 2, num_features)  
y = np.random.rand(num_samples) 




###### Create an rTPNN Model

input_layer = Input(input_shape=(2, num_features, ))

**rtpnn_layer = rTPNN()(input_layer)**

fullyconnected_layer = Dense(num_features, activation='relu')(rtpnn_layer)

output_layer = Dense(1, activation='relu')(fullyconnected_layer)


rTPNN_model = Model(inputs=[input_layer], outputs=[output_layer])

rTPNN_model.compile(optimizer='adam', loss='mse')  


###### Train the Model

rTPNN_model.fit(x, y, epochs=10, batch_size=20, verbose=0)  


###### Make Prediction


prediction = rTPNN_model.predict(x) 

## Citation Request 
The rTPNN as well as its application on multi-sensor fire detection has been published as a journal paper which is entitled as "Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection" on IEEE Access. If you use rTPNN or the content of this repository, please cite our following paper as follows: 

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
