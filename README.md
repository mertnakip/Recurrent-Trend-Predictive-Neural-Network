# Recurrent Trend Predictive Neural Network


![Alt text](Figures/rTPNN-FireDetector.jpg?raw=true "Title")

This repository contains the implementation of the Recurrent Trend Predictive Neural Network (rTPNN) model as a Keras layer. In addition, it also contains an application of rTPNN for the multi-sensor fire detection in the folder FireDetection_via_rTPNN.

You may find the more detailed explanation of the methodology as well as the results in our publication at https://ieeexplore.ieee.org/document/9451553.

Note that it is an particular implementation of rTPNN, and it may be implemented in different ways.

## Inputs for rTPNN Layer

Provide input array "x" as shown in the following figure. 

![Alt text](Figures/Tensor.jpg?raw=true "Title")


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

input_layer = Input(input_shape=(2, num_features,))

**rtpnn_layer = rTPNN()(input_layer)**

fullyconnected_layer = Dense(num_features, activation='relu')(rtpnn_layer)

output_layer = Dense(1, activation='relu')(fullyconnected_layer)


rTPNN_model = Model(inputs=[input_layer], outputs=[output_layer])

rTPNN_model.compile(optimizer='adam', loss='mse')  


###### Train the Model

rTPNN_model.fit(x, y, epochs=10, batch_size=20, verbose=0) 

'''
batch_size determines the time interval for the update of recurrence. "The last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch." [https://keras.io/api/layers/recurrent_layers/simple_rnn/]
'''

###### Make Prediction


prediction = rTPNN_model.predict(x, batch_size=1) 



## Citation Request 
The rTPNN as well as its application on multi-sensor fire detection has been published as a journal paper which is entitled as "Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection" on IEEE Access. If you use rTPNN or the content of this repository, please cite our following paper (along with the repository citation) as follows: 

@ARTICLE{nakip2021rTPNN,  
  author={Nakip, Mert and Güzeliş, Cüneyt and Yildiz, Osman},  
  journal={IEEE Access},  
  title={Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection},  
  year={2021},  
  volume={9},  
  number={},  
  pages={84204-84216},  
  doi={10.1109/ACCESS.2021.3087736}  
  }
