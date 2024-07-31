# Recurrent Trend Predictive Neural Network


![Alt text](Figures/rTPNN-FireDetector.jpg?raw=true "Title")

This repository contains the implementation of the Recurrent Trend Predictive Neural Network (rTPNN) model as a Keras layer. In addition, it also contains an application of rTPNN for multi-sensor fire detection in the folder FireDetection_via_rTPNN.

You may find a more detailed explanation of the methodology as well as the results in our publication at https://ieeexplore.ieee.org/document/9451553.

Note that it is a particular implementation of rTPNN, and it may be implemented in different ways.

## Inputs for rTPNN Layer

Provide input array "x" as shown in the following figure. 

![Alt text](Figures/Tensor.PNG?raw=true "Title")


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


## Applications of rTPNN


Fire Detection: https://github.com/mertnakip/Recurrent-Trend-Predictive-Neural-Network/tree/main/FireDetection_via_rTPNN 

Energy Management and Forecasting: https://github.com/mertnakip/Recurrent-Trend-Predictive-Neural-Network/tree/rtpnn_sef 


## Citation Request 
The rTPNN, as well as its application on multi-sensor fire detection, has been published as a journal paper which is entitled "Recurrent Trend Predictive Neural Network for Multi-Sensor Fire Detection" in IEEE Access. If you use rTPNN or the content of this repository, please cite our following paper (along with the repository citation) as follows: 

```
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
  ```

## Additional References 

###### rTPNN-FES Architecture for Energy Management

```
@article{NAKIP_rTPNN_FES,
  title = {Renewable energy management in smart home environment via forecast embedded scheduling based on Recurrent Trend Predictive Neural Network},
  author={Nak{\i}p, Mert and {\c{C}}opur, Onur and Biyik, Emrah and G{\"u}zeli{\c{s}}, C{\"u}neyt},
  journal = {Applied Energy},
  volume = {340},
  pages = {121014},
  year = {2023},
  issn = {0306-2619},
  doi = {https://doi.org/10.1016/j.apenergy.2023.121014},
  url = {https://www.sciencedirect.com/science/article/pii/S0306261923003781}
}
```

###### rTPNN with Online Learning for E-Nose
```
@ARTICLE{bulucu_ertpnn,
  author={Bulucu, Pervіn and Nakip, Mert and Güzelіș, Cüneyt},
  journal={IEEE Access}, 
  title={Multi-Sensor E-Nose Based on Online Transfer Learning Trend Predictive Neural Network}, 
  year={2024},
  volume={12},
  number={},
  pages={71442-71452},
  keywords={Market research;Transfer learning;Long short term memory;Feature extraction;Convolutional neural networks;Quality control;Electronic noses;Multisensor systems;Neural networks;E-Nose;trend prediction;multi-sensor;recurrent trend predictive neural network;online learning},
  doi={10.1109/ACCESS.2024.3401569}}
```
