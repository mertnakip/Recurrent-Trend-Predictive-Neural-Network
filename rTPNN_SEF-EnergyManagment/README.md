# Recurrent Trend Predictive Neural Network based Forecast Embedded Scheduling


![Alt text](figures/rtpnn_fes.jpg?raw=true "Title")

This repository contains the implementation of the Recurrent Trend Predictive Neural Network based Forecast Embedded Scheduling (rTPNN-FES) model.  rTPNN-FES is a novel neural network architecture that simultaneously forecasts renewable energy generation and schedules household appliances. By its embedded structure, rTPNN-FES eliminates the utilization of separate algorithms for forecasting and scheduling and generates a schedule that is robust against forecasting errors. 

You may find a more detailed explanation of the methodology as well as the results in our publication at https://www.sciencedirect.com/science/article/pii/S0306261923003781.


## Forecasting Layer


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/146jvz5zUx1DiELXhfQiIBk1kWFOXxLtz?usp=sharing)


Forecasting Layer is responsible for forecasting the power generation within the architecture of rTPNN-FES. For each slot $s$ in the scheduling window, rTPNN-FES forecasts the renewable energy generation $\hat{g}^{m_s}$ based on the collection of the past feature values for two periods, $z^{m_s - 2 \tau_f}_f, z^{m_s -\tau_f}_f$, as well as the past generation for two periods $\{g^{m_s - 2\tau_0}, g^{m_s - \tau_0}\}$.


## Scheduling Layer


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eH0M75-jUh4wZOncQnCAn-Q4JmfCHWWd?usp=sharing)


The Scheduling Layer consists of $N$ parallel softmax layers, each responsible for generating a schedule for a single device's start time. Since this layer is cascaded behind the Forecasting Layer, each device $n$ is scheduled to be started at each slot $s$ based on the output of the Forecasting Layer $\hat{g}^{m_s}$ as well as the system parameters $c_{(n,s)}$, $E_n$, $B$, $B_{max}$ and $\Theta$ for this device $n$ and this slot $s$.  



## Citation

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

