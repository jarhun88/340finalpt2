import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import os
import glob
from neural_net import NeuralNet

class CSV_Processor():

    def __init__(self, knnMode=False):
        self.knnMode = knnMode

    def process_csv(self, Xpath, yPath):
        # pre-processing X train data
        # ROOT_DIR = os.path.abspath(os.curdir)
        # csvtrainingDir = ROOT_DIR
        all_X_files = glob.glob(Xpath) 
        # print(all_X_files)

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
                    if 'y' in col:
                        print(col)
                        flag = 0
                    if 'x' in col: 
                        print(col)
                        continue
                    else:
                        df.drop(col, axis=1, inplace=True)
                else:
                    df.drop(col, axis=1, inplace=True)

            print(df)        

                # flag--
                # if flag < 0: 
                #     df.drop(col, axis=1, inplace=True)

                # if 'agent' in col:
                #     flag = 2
                # if 'time' in col or 'id' in col or 'type' in col:
                #     df.drop(col, axis=1, inplace=True)

                # if 'role' in col:
                #     # convert 'role' values to integers
                #     for idx in range(11):
                #         if df.loc[idx, col] == 'agent':
                #             df.loc[idx, col] = 1
                #         else:
                #             df.loc[idx, col] = 0
                
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