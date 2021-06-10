from rTPNN import rTPNN
from ProcessData import ProcessData
from CrossValidation import CrossValidation
import math
import numpy as np

def print_results(training_results, test_results, model_name):
    conf_mat = training_results[1]
    folder = ''
    print_text1 = 'Mean Training: ' + str(np.mean(training_results[0])) + ' STD Training: '+ str(np.std(training_results[0])) + ' Mean Test: ' + str(np.mean(test_results[0])) + ' STD Test: '+ str(np.std(test_results[0])) 
    print_text2 = ' True Positive: ' + str(np.mean(np.array(conf_mat)[:, 1, 1])) + ' True Negative: ' + str(np.mean(np.array(conf_mat)[:, 0, 0])) 
    print_text3 = ' False Positive: ' + str(np.mean(np.array(conf_mat)[:, 0, 1])) + ' False Negative: ' + str(np.mean(np.array(conf_mat)[:, 1, 0]))
    print_text4 = ' STD Positives: ' + str(np.std(np.array(conf_mat)[:, 1, 1])) + ' STD Negatives: ' + str(np.std(np.array(conf_mat)[:, 0, 0])) + ' \n'
    
    print_text = print_text1 + print_text2 + print_text3 + print_text4
    print(print_text)

    with open(folder+model_name+"_CV.txt","a+") as f:
      f.write(print_text)

# Read and shape data
print('\n ============ DATASET ============')
 
NIST_data_file = 'Dataset'

data_processor = ProcessData(NIST_data_file, [])
sensors_considered = ["\'TCB_1 \'", "\'SMB_1 \'", "\'GASB_1\'", "\'GASB_3\'", "\'GASB_5\'"]#data_processor.find_common_sensors()
data_processor.selected_columns = sensors_considered 
sensors_considered_int = np.array(range(5))

failed_exp = [2, 15, 17]
bedroom_exp = np.array([3, 4, 5, 6, 21, 22, 23, 24, 25])[:, np.newaxis] #

data_processor.read_data(bedroom_exp)
x, y, relative_time = data_processor.shape_to_input_output(sensors_considered_int)
all_indices = range(y.shape[0])

# ====================================
num_sensors = len(sensors_considered)
predictor_arch = [num_sensors*2, num_sensors, math.ceil(num_sensors/2)]

# =========== rTPNN Model ==============
print('\n ============ rTPNN Model ============')
r_tpnn = rTPNN(num_sensors, predictor_arch, activation_name='sigmoid')
rTPNN_model = r_tpnn.model

CV_rtpnn = CrossValidation(x, y, relative_time, all_indices, sensors_considered_int, model_name='rTPNN', model_is_rTPNN=True)
training_results_rtpnn, test_results_rtpnn = CV_rtpnn.run(rTPNN_model)

print_results(training_results_rtpnn, test_results_rtpnn, 'rTPNN')


