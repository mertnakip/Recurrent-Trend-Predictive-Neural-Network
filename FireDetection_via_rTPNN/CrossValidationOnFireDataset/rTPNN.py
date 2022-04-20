from keras import Model
from keras.layers import Dense, Input, Lambda, SimpleRNN, Concatenate, Subtract, Reshape
import keras.backend as K

class rTPNN:
    def __init__(self, num_sensors, predictor_arch, activation_name='tanh'):
        self.num_sensors = num_sensors
        self.predictor_arch = predictor_arch
        self.activation_name = activation_name
        self.model = self.create_model()

    def slice_0(self, x):
        return x[:, 0:1, :]
    def slice_1(self, x):
        return x[:, 1:2, :]

    def slice_output_shape(self, input_shape):
        shape = [input_shape[0], 1, 1]
        return tuple(shape)

    def SDPs(self):
        inputs_SDP = []
        trend_predictor = []
        level_predictor = []
        outputs_SDP = []
        for i in range(self.num_sensors):
            inputs_SDP.append(Input(shape=(2, 1, ))) #Input1 is x_{k} and Input2 is x_{k-1}
            x_k = Lambda(self.slice_0, output_shape=self.slice_output_shape)(inputs_SDP[-1])
            x_k1 = Lambda(self.slice_1, output_shape=self.slice_output_shape)(inputs_SDP[-1])

            #Trend Predictor
            trend_predictor.append(Subtract()([x_k, x_k1]))
            trend_predictor[-1] = SimpleRNN(1, activation=None, use_bias=False)(trend_predictor[-1])

            #Level Predictor
            level_predictor.append(x_k)
            level_predictor[-1] = SimpleRNN(1, activation=None, use_bias=False)(level_predictor[-1])

            #SDP Dense Neuron
            outputs_SDP.append(Concatenate()([trend_predictor[-1], level_predictor[-1], Reshape((1,))(x_k)]))
        return inputs_SDP, outputs_SDP

    def create_model(self):
        #SDPs
        inputs_SDP, outputs_SDP = self.SDPs()

        #Fire Predictor
        if self.num_sensors == 1:
          input_fully_connected = outputs_SDP[-1]
        else:
            input_fully_connected = Concatenate()(outputs_SDP)
        output = Dense(self.predictor_arch[0], activation=self.activation_name)(input_fully_connected)
        for n in self.predictor_arch[1:]:
            output = Dense(n, activation=self.activation_name)(output)
        output = Dense(1, activation='sigmoid')(output)

        return Model(inputs=inputs_SDP, outputs=output)
