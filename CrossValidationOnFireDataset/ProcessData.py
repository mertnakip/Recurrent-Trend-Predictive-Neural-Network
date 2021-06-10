import numpy as np
import pandas as pd
import os
import scipy.io as spio
import math
from random import sample

class ProcessData:
    def __init__(self, data_set_directory, columns_to_use):
        self.directory = data_set_directory
        self.selected_columns = columns_to_use
        self.datasets = []
        self.all_common_sensors = []
        #self.read_data()

    def find_common_sensors(self):
        files = os.listdir(self.directory)
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(self.directory+'/'+file)
                sensors = list(df.columns[1:])
                if len(self.all_common_sensors) == 0:
                    self.all_common_sensors = sensors
                else:
                    self.all_common_sensors = list(set(self.all_common_sensors) & set(sensors))
        sensors = []
        for sensor in self.all_common_sensors:
            sensors.append([])
            for file in files:
                if file.endswith(".csv"):
                    sensors[-1].append(sensor)
        self.all_common_sensors = sensors
        return self.all_common_sensors

    def read_data(self, considered_files):
        files = os.listdir(self.directory)
        count = 0
        for file in files:
            if file.endswith(".csv") and count in considered_files:
                df = pd.read_csv(self.directory+'/'+file)
                unselected_columns = list(df.columns)
                for x in [unselected_columns[0]] + self.selected_columns:
                    #print(file, x)
                    unselected_columns.remove(x)
                df = df.drop(columns=unselected_columns)
                df['fire_label'] = np.ones(df.iloc[:, 0].shape)
                df.iloc[np.where(df.iloc[:, 0] < 10), -1] = 0
                data = self.process_data(df)
                self.datasets.append(data)
            if file.endswith(".csv"):
                count += 1

    def process_data(self, df):
        num_sensors = len(df.columns)-2
        x = []
        for i in range(1, num_sensors + 1):
            x.append(np.array([np.array(df.iloc[1:, i:i + 1]), np.array(df.iloc[:-1, i:i + 1])])[:, :, 0].T[:, :, np.newaxis])

        y = np.array(df.iloc[1:, -1:])
        time = np.array(df.iloc[1:, 0:1])
        return [x, y, time]

    def shape_to_input_output(self, sensors):
        x = []

        for j in sensors:
            count = 0
            for i in range(len(self.datasets)):
                if count == 0:
                    x.append(self.datasets[i][0][j])
                    y = self.datasets[i][1]
                    time = self.datasets[i][2]
                else:
                    x[-1] = np.concatenate((x[-1], self.datasets[i][0][j]), axis=0)
                    y = np.concatenate((y, self.datasets[i][1]), axis=0)
                    time = np.concatenate((time, self.datasets[i][2]), axis=0)
                count += 1
            x[-1] = x[-1]/np.max(x[-1])
        
        rand_samples = sample(range(y.shape[0]), y.shape[0]) 
        #spio.savemat('rand_indices.mat', {'indices': rand_samples})   
        '''
        # To load the previously saved random indices
        rand_samples = spio.loadmat('rand_indices.mat')
        rand_samples = list(rand_samples['indices'][0])
        '''
        y = y[rand_samples, :]
        time = time[rand_samples, :]
        for j in sensors:
          x[j] = x[j][rand_samples, :, :]
        
        return x, y, time
