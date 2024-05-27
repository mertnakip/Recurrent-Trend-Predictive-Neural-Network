# Multi-Sensor E-Nose based on Trend Predictive Neural Network with Online Transfer Learning
![Alt text](figures/e-rtpnn.png)

This repository contains the implementation of the the e-rTPNN decision system. This model leverages the Recurrent Trend Predictive Neural Network (rTPNN) combined with online transfer learning for electronic nose (e-Nose) applications.

The detailed explanation of the architecture and comparative results on wine and COPD e-Nose datasets can be found in our published article 
[link](link).

## Prepare the datasets
### Wine Dataset
Download the wine dataset from https://data.mendeley.com/datasets/vpc887d53s/3 and put it under wine_dataset folder in google drive

Run the following code, which will create Wine_Dataset_All folder and file_list.npy files

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12rwM16-W9wisFp0GtgZUDR4CVVdWLf_H?usp=sharing)

### COPD Dataset
Download the COPD dataset from [https://data.mendeley.com/datasets/vpc887d53s/3](https://data.mendeley.com/datasets/h5pcn99zw4/4) and put it under COPD_dataset folder in google drive

Run the following code, which will create COPD_Dataset_All folder and file_list.npy files

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tJ-gwzBhwUCaw3vxui_xX_DjtJFDL2mZ?usp=sharing)

## Train e-rTPNN
For online training to be performed after the offline training, Set the online_only array:

`online_only = [False, False, False, False, False]`

To perform the online training only, Set the online_only array:

`online_only = [True, True, True, True, True]`

### Train on Wine Dataset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j29qxD0sQcV9sVq7zuBrtG4o2k_kJFz1?usp=sharing)

### Train e-rtpnn on COPD Dataset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bikva_wEF_2qyhIpqVzfGwWw9986y08x?usp=sharing)

## Citation
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
