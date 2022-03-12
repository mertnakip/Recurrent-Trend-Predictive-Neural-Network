import numpy as np
from ConvertDataForRTPNN import convert_data_for_rTPNN
from rTPNN import rTPNN


T = 100
I = 5

x = np.random.rand(T, 2, I)
y = np.random.rand(T)

x_list = convert_data_for_rTPNN(x)


r_tpnn = rTPNN(I, [I*2, I, np.ceil(I/2)], activation_name='sigmoid')
rTPNN_model = r_tpnn.model
rTPNN_model.compile(optimizer='adam', loss='mse')
rTPNN_model.fit(x_list, y, epochs=10, batch_size=20, verbose=0)
prediction = rTPNN_model.predict(x_list)
