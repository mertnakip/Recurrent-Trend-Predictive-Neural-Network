import numpy as np
 
class Metrics:
    def calc_conf_mat(self, actual, predicted):

        positive_indices = np.where(actual == 1)[0]
        negative_indices = np.where(actual == 0)[0]

        true_positive = np.sum((actual == predicted)[positive_indices]) / positive_indices.shape[0]
        true_negative = np.sum((actual == predicted)[negative_indices]) / negative_indices.shape[0]
        false_positive = np.sum((actual < predicted)[negative_indices]) / negative_indices.shape[0]
        false_negative = np.sum((actual > predicted)[positive_indices]) / positive_indices.shape[0]

        accuracy = np.sum(actual == predicted) / actual.shape[0]

        conf_mat = np.zeros((2, 2))
        conf_mat[0, 0] = true_negative
        conf_mat[0, 1] = false_positive
        conf_mat[1, 0] = false_negative
        conf_mat[1, 1] = true_positive

        return accuracy, conf_mat

    def calc_alarming_time(self, actual, predicted, relative_time):
        #fire_first_index = np.where(actual == 1)[0][0]
        predicted = predicted * actual
        try:
            predicted_fire_first_index = np.where(predicted == 1)[0][0]
        except:
            predicted_fire_first_index = predicted.shape[0]-1
        alarming_time = relative_time[predicted_fire_first_index] - 10 #- relative_time[fire_first_index]
        return alarming_time
