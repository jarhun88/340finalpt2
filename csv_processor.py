import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import os
import glob
import re
from neural_net import NeuralNet


class CSV_Processor():

    def process_csv(self, Xpath, yPath):
        # pre-processing X train data
        # ROOT_DIR = os.path.abspath(os.curdir)
        # csvtrainingDir = ROOT_DIR
        all_X_files = os.listdir(Xpath)
        all_X_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        # print(all_X_files)

        main_X_file = []
        for file in all_X_files:
            df = pd.read_csv(Xpath + file, index_col=None, header=0)
            # print(file)
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

        # convert main_X_file to np.array

        # print(main_X_array)

        # pre-processing Y train data
        all_y_files = os.listdir(yPath)
        all_y_files.sort(key=lambda f: int(re.sub('\D', '', f)))
        # print(all_y_files)

        main_y_file = []
        for i in range(len(all_y_files)):
            df = pd.read_csv(yPath + all_y_files[i], index_col=None, header=0)
            # print(file)
            for col in df.columns:
                if 'time' in col:
                    df.drop(col, axis=1, inplace=True)

            array = df.to_numpy(dtype='float64')
            flattened_array = array.flatten()
            if flattened_array.shape[0] != 60:
                # dont add and also delete the associating x val
                print("happened")
                del main_X_file[i]
                continue
            main_y_file.append(flattened_array)


        # convert main_files to np.array
        main_X_array = np.array(main_X_file, dtype='float64')
        main_y_array = np.array(main_y_file, dtype='float64')
        # print(main_y_array)


        return main_X_array, main_y_array
