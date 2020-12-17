import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import time
import os
import glob
import re
from neural_net import NeuralNet

ROOT_DIR = os.path.abspath(os.curdir)

sample_submission = ROOT_DIR + "\sample_submission.csv"

class CSV_Processor():

    def __init__(self, knnMode=False):
        self.knnMode = knnMode

    def process_output_csv(self, path):
        df = pd.read_csv(path, index_col=None, header=0)
        return df

    def to_kaggle_csv(self, y):
        retcsv = y.flatten()
        retcsv2 = pd.DataFrame(retcsv, columns=["location"])
        retcsv2.to_csv("y_pred.csv", index=False)

        df = self.process_output_csv(sample_submission)
        df.drop(df.columns[1], axis=1, inplace=True)
        df2 = df.join(retcsv2)
        df2.to_csv('y_output.csv', index=False)

    def process_test_csv(self, Xpath):
        all_X_files = glob.glob(Xpath)
        # print(all_X_files)

        main_X_file = []
        for file in all_X_files:
            df = pd.read_csv(file, index_col=None, header=0)
            flag = 0
            for col in df.columns:
                if 'role' in col:
                    if isinstance(df.loc[0, col], float):
                        df.drop(col, axis=1, inplace=True)
                    elif 'agent' in df.loc[0, col]:
                        # print(df.loc[0, col])
                        flag = 1
                        df.drop(col, axis=1, inplace=True)
                    else:
                        df.drop(col, axis=1, inplace=True)
                elif flag == 1:
                    if 'type' in col:
                        df.drop(col, axis=1, inplace=True)
                    elif 'y' in col:
                        # print(col)
                        flag = 0
                        continue
                    elif 'x' in col:
                        # print(col)
                        continue
                    else:
                        df.drop(col, axis=1, inplace=True)
                else:
                    df.drop(col, axis=1, inplace=True)

            array = df.to_numpy()
            flattened_array = array.flatten()
            main_X_file.append(flattened_array)

        # convert main_X_file to np.array
        main_X_array = np.array(main_X_file)
        return main_X_array

    def process_csv(self, Xpath, yPath):
        all_X_files = glob.glob(Xpath)

        main_X_file = []
        for file in all_X_files:
            df = pd.read_csv(file, index_col=None, header=0)
            # print(file)
            flag = 0
            for col in df.columns:
                if 'role' in col:
                    if isinstance(df.loc[0, col], float):
                        df.drop(col, axis=1, inplace=True)
                    elif 'agent' in df.loc[0, col]:
                        # print(df.loc[0, col])
                        flag = 1
                        df.drop(col, axis=1, inplace=True)
                    else:
                        df.drop(col, axis=1, inplace=True)
                elif flag == 1:
                    if 'type' in col:
                        df.drop(col, axis=1, inplace=True)
                    elif 'y' in col:
                        # print(col)
                        flag = 0
                        continue
                    elif 'x' in col:
                        # print(col)
                        continue
                    else:
                        df.drop(col, axis=1, inplace=True)
                else:
                    df.drop(col, axis=1, inplace=True)

            array = df.to_numpy()
            flattened_array = array.flatten()
            main_X_file.append(flattened_array)

        # convert main_X_file to np.array
        main_X_array = np.array(main_X_file)
        # print(main_X_array)

        # pre-processing Y train data
        all_y_files = glob.glob(yPath)
        # print(all_y_files)

        main_y_file = []
        for file in all_y_files:
            df = pd.read_csv(file, index_col=None, header=0)
            # print(file)
            for col in df.columns:
                if 'time' in col:
                    df.drop(col, axis=1, inplace=True)

            array = df.to_numpy()
            flattened_array = array.flatten()
            main_y_file.append(flattened_array)

        # convert main_y_file to np.array
        main_y_array = np.array(main_y_file)
        # print(main_y_array)

        return main_X_array, main_y_array