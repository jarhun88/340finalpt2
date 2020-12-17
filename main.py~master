import numpy as np
import pandas as pd
import time
import os
from neural_net import NeuralNet
from csv_processor import CSV_Processor

# pre-processing X train data
ROOT_DIR = os.path.abspath(os.curdir)
#for mac
# all_X_train_files = ROOT_DIR + "data/train/X/*.csv"
# all_Y_train_files = ROOT_DIR + "/data/train/y/*.csv"
# all_X_val_files = ROOT_DIR + "/data/val/X/*.csv"
# all_Y_val_files = ROOT_DIR + "/data/val/y/*.csv"

# for windows
all_X_train_files = ROOT_DIR + "\\data\\train\\X\\"
all_Y_train_files = ROOT_DIR + "\\data\\train\\y\\"
all_X_val_files = ROOT_DIR + "\\data\\val\\X\\"
all_Y_val_files = ROOT_DIR + "\\data\\val\\y\\"
all_X_test_files = ROOT_DIR + "\\data\\test\\X\\"
sample_submission = ROOT_DIR + "\\data\\sample_submission.csv"

csv_processor = CSV_Processor()

Xtrain, ytrain = csv_processor.process_csv(all_X_train_files, all_Y_train_files)
# Xval, yval = csv_processor.process_csv(all_X_val_files, all_Y_val_files)
Xtest = csv_processor.process_test_csv(all_X_test_files)

best_rmse = 1000
best_index = 0
# for i in range(100):`
hidden_layer_sizes = [76]
model = NeuralNet(hidden_layer_sizes)
t = time.time()
model.fit(Xtrain, ytrain)
print("Fitting took %d seconds" % (time.time()-t))

# Compute test error
yhat = model.predict(Xtest)

# rmse = csv_processor.find_rmse(yhat, yval)
# print(rmse, i)
# if rmse < best_rmse:
#     best_rmse = rmse
#     best_index = i
#
# print(best_rmse)
# print(best_index)

# export as csv
csv_processor.to_kaggle_csv(yhat)



