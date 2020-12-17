import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import os
import glob
import re
from neural_net import NeuralNet
from sklearn import metrics

ROOT_DIR = os.path.abspath(os.curdir)
sample_submission = ROOT_DIR + "\\data\\sample_submission.csv"

# csv processor for MLP model
class CSV_Processor():

    # converts numpy ndarray to csv file for submission
    def to_kaggle_csv(self, y):
        retcsv = y.flatten()
        retcsv2 = pd.DataFrame(retcsv, columns=["location"])
        retcsv2.to_csv("y_pred.csv", index=False)

        df = self.process_output_csv(sample_submission)
        df.drop(df.columns[1], axis=1, inplace=True)
        df2 = df.join(retcsv2)
        df2.to_csv('y_output.csv', index=False)

    # read file path into pd dataframe object
    def process_output_csv(self, path):
        df = pd.read_csv(path, index_col=None, header=0)
        return df

    # convert path into nd array with specific features removed
    def process_test_csv(self, path):
        all_X_files = os.listdir(path)
        all_X_files.sort(key=lambda f: int(re.sub('\D', '', f)))

        main_X_file = []
        for file in all_X_files:
            df = pd.read_csv(path + file, index_col=None, header=0)
            for col in df.columns:
                # removing time, id and type feature columns from csv file
                if 'time' in col or 'id' in col or 'type' in col:
                    df.drop(col, axis=1, inplace=True)
                # convert 'role' values to integers
                if 'role' in col:
                    for idx in range(11):
                        if df.loc[idx, col] == 'agent':
                            df.loc[idx, col] = 1
                        else:
                            df.loc[idx, col] = 0
            array = df.to_numpy(dtype='float64')
            flattened_array = array.flatten()
            main_X_file.append(flattened_array)
        main_X_array = np.array(main_X_file, dtype='float64')
        return main_X_array

    # convert Xpath and ypath into ndarray with specific features removed
    def process_csv(self, Xpath, yPath):
        # pre-processing X train data
        all_X_files = os.listdir(Xpath)
        all_X_files.sort(key=lambda f: int(re.sub('\D', '', f)))

        main_X_file = []
        for file in all_X_files:
            df = pd.read_csv(Xpath + file, index_col=None, header=0)
            for col in df.columns:
                if 'time' in col or 'id' in col or 'type' in col:
                    df.drop(col, axis=1, inplace=True)

                if 'role' in col:
                    # convert 'role' values to integers
                    for idx in range(11):
                        if df.loc[idx, col] == 'agent':
                            df.loc[idx, col] = 1
                        else:
                            df.loc[idx, col] = 0
            array = df.to_numpy(dtype='float64')
            flattened_array = array.flatten()
            main_X_file.append(flattened_array)

        # pre-processing Y train data
        all_y_files = os.listdir(yPath)
        all_y_files.sort(key=lambda f: int(re.sub('\D', '', f)))

        main_y_file = []
        for i in range(len(all_y_files)):
            df = pd.read_csv(yPath + all_y_files[i], index_col=None, header=0)
            for col in df.columns:
                if 'time' in col:
                    df.drop(col, axis=1, inplace=True)

            array = df.to_numpy(dtype='float64')
            flattened_array = array.flatten()
            if flattened_array.shape[0] != 60:
                # dont add and also delete the associating x val
                del main_X_file[i]
                continue
            main_y_file.append(flattened_array)

        # convert main_files to np.array
        main_X_array = np.array(main_X_file, dtype='float64')
        main_y_array = np.array(main_y_file, dtype='float64')

        return main_X_array, main_y_array

    # rmse function for calculating kaggle score
    def find_rmse(self, y_pred_csv, y_test_csv):
        y_pred = []
        y_test = []

        for row in y_pred_csv:
            # if column number is even this is a x
            for i in range(len(row) - 1):
                if i % 2 == 0:
                    point = [row[i], row[i + 1]]
                    y_pred.append(point)
        y_pred = np.array(y_pred)

        for row in y_test_csv:
            # if column number is even this is a x
            for i in range(len(row) - 1):
                if i % 2 == 0:
                    point = [row[i], row[i + 1]]
                    y_test.append(point)
        y_test = np.array(y_test)

        mean = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mean / 600)
        return rmse
