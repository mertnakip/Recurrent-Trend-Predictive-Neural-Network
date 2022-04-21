from keras.layers import Lambda, SimpleRNN, Concatenate, Subtract, Reshape

class rTPNN:        
    def __call__(self, inputs):
        """
        Inputs: 3D array with dimensions (Sample Size X 2 X Number of Features)
        """
        num_features = inputs.shape[-1]
        trend_predictor = []
        level_predictor = []
        outputs_DP = []
        for i in range(num_features):
            Slicing_layer1 = Lambda(lambda x, start, end: x[:, 0:1, start:end], output_shape=(1, 1,), name="slicing_layer1_"+str(i))
            Slicing_layer1.arguments = {'start': i, 'end': (i+1)}
            x_k = Slicing_layer1(inputs)

            Slicing_layer2 = Lambda(lambda x, start, end: x[:, 1:2, start:end], output_shape=(1, 1,), name="slicing_layer2_"+str(i))
            Slicing_layer2.arguments = {'start': i, 'end': (i+1)}
            x_k1 = Slicing_layer2(inputs)

            #Trend Predictor
            trend_predictor.append(Subtract()([x_k, x_k1]))
            trend_predictor[-1] = SimpleRNN(1, activation=None, use_bias=False, stateful=True)(trend_predictor[-1])

            #Level Predictor
            level_predictor.append(x_k)
            level_predictor[-1] = SimpleRNN(1, activation=None, use_bias=False, stateful=True)(level_predictor[-1])

            #DP Dense Neuron
            outputs_DP.append(Concatenate()([trend_predictor[-1], level_predictor[-1], Reshape((1,))(x_k)]))
        return Concatenate()(outputs_DP)

