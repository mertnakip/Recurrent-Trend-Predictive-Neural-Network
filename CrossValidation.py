from sklearn.model_selection import KFold
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import time
from keras.metrics import BinaryAccuracy
from keras.optimizers import Adam
from Metrics import Metrics

class CrossValidation:
    def __init__(self, x, y, relative_time, avaliable_dataset_indices, sensors, model_name, model_is_rTPNN=True, model_is_LSTM=False, model_is_SVM=False, num_fold=10):
        self.kfold = KFold(n_splits=num_fold, shuffle=False)
        self.x = x
        self.y = y
        self.relative_time = relative_time
        self.avaliable_dataset_indices = avaliable_dataset_indices
        self.es = EarlyStopping(monitor="loss", patience=75, verbose=1, mode="min", restore_best_weights=True)
        self.model_is_rTPNN = model_is_rTPNN
        self.model_is_LSTM =model_is_LSTM
        self.model_is_SVM = model_is_SVM
        self.sensors = sensors
        self.metrics = Metrics()
        self.threshold = 0.5
        self.model_name = model_name

    def run(self, model):
        accuracy, conf_mat, alarming_time, execution_time = [], [], [], []
        accuracy_train, conf_mat_train, alarming_time_train, time_train = [], [], [], []
        cnt = 0
        if not self.model_is_SVM:
            model.save_weights('initial_weights.h5')
        for train_indices, test_indices in self.kfold.split(self.avaliable_dataset_indices):
          cnt += 1
          if cnt > 0:
            global_train_ind = train_indices #self.avaliable_dataset_indices[train_indices].flatten()
            global_test_ind = test_indices #self.avaliable_dataset_indices[test_indices].flatten()

            # TRAIN
            if not self.model_is_SVM:
                model.load_weights('initial_weights.h5')
                opt = Adam(learning_rate=0.005)
                model.compile(optimizer='adam', loss='mse', metrics=[BinaryAccuracy()])

            model, train_time = self.train(model, global_train_ind, global_test_ind)
            self.threshold = self.select_threshold(model, global_train_ind)

            accuracy_cv, conf_mat_cv, alarming_time_cv, execution_time_cv = self.test(model, global_train_ind)
            accuracy_train.append(accuracy_cv)
            conf_mat_train.append(conf_mat_cv) 
            alarming_time_train.append(alarming_time_cv)
            time_train.append(train_time)

            # TEST
            accuracy_cv, conf_mat_cv, alarming_time_cv, execution_time_cv = self.test(model, global_test_ind)
            accuracy.append(accuracy_cv)
            conf_mat.append(conf_mat_cv)
            alarming_time.append(alarming_time_cv)
            execution_time.append(execution_time_cv)

            folder = ''
            print_text = 'CV: ' + str(cnt) + ' Train Accuracy: ' + str(np.mean(accuracy_train[-1])) + ' Mean Accuracy: ' + str(np.mean(accuracy_cv)) + ' False Positive: ' + str(np.mean(conf_mat_cv[0, 1])) + ' False Negative: ' + str(np.mean(conf_mat_cv[1, 0])) + ' Mean Alarm Time: ' + str(np.mean(alarming_time_cv)) + ' \n'
            print(print_text)

            with open(folder+self.model_name+"_CV.txt","a+") as f:
              f.write(print_text)

        return [accuracy_train, conf_mat_train, alarming_time_train, time_train], [accuracy, conf_mat, alarming_time, execution_time]

    def get_input_output(self, indices):
        inputs = []
        outputs = self.y[indices, :]
        times = self.relative_time[indices, :]
        for j in self.sensors:
            inputs.append(self.x[j][indices, :, :])

        if not self.model_is_rTPNN:
            if self.model_is_LSTM:
                inputs = np.concatenate([np.array(inputs)[:, :, 0, 0].T[:, :, np.newaxis], np.array(inputs)[:, :, 1, 0].T[:, :, np.newaxis]], axis=-1)
            else:
                inputs = np.array(inputs)[:, :, 0, 0].T
            if self.model_is_SVM:
                outputs = outputs[:, 0]
        return inputs, outputs, times

    def train(self, model, train_indices, test_indices):
        x_train, y_train, times_train = self.get_input_output(train_indices)

        start_train_time = time.time()
        if self.model_is_SVM:
            model.fit(x_train, y_train) 
        else:
            model.fit(x_train, y_train, epochs=2000, batch_size=20, verbose=0, callbacks=[self.es]) 
        train_time = time.time() - start_train_time
        return model, train_time

    def convert_to_binary(self, p):
        p[p >= self.threshold] = 1
        p[p < self.threshold] = 0
        #p = np.array([p[i][0] for i in range(3)]+[np.median(p[i-3:i]) for i in range(3, p.shape[0])])[:, np.newaxis]
        return p

    def select_threshold(self, model, test_indices):
        x, d, relative_time = self.get_input_output(test_indices)
        y = model.predict(x)
        if self.model_is_SVM:
            y = y[:, np.newaxis]
            d = d[:, np.newaxis]

        acc = []
        threshold_vals = np.array(range(20)) / 20
        for threshold in threshold_vals:
            self.threshold = threshold
            y_pred = np.copy(y)
            y_binary = self.convert_to_binary(y_pred)
            acc.append(accuracy_score(y_binary, d))
        best_threshold = threshold_vals[np.argmax(np.array(acc))]
        return best_threshold

    def test(self, model, test_indices):
        x_test, y_test, relative_time = self.get_input_output(test_indices)

        start_exacution_time = time.time()
        prediction = model.predict(x_test)

        execution_time = time.time()-start_exacution_time
        if self.model_is_SVM:
            y_test = y_test[:, np.newaxis]
            prediction = prediction[:, np.newaxis]

        prediction = self.convert_to_binary(prediction)


        alarming_time = self.metrics.calc_alarming_time(y_test, prediction, relative_time)
        accuracy, conf_mat = self.metrics.calc_conf_mat(y_test, prediction)

        return accuracy, conf_mat, alarming_time, execution_time