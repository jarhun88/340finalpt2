import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import os
import glob
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

csv_processor = CSV_Processor()

Xtrain, ytrain = csv_processor.process_csv(all_X_train_files, all_Y_train_files)
Xval, yval = csv_processor.process_csv(all_X_val_files, all_Y_val_files)

# print(Xtrain)
# print(ytrain)
# print(Xval)
# print(yval)

hidden_layer_sizes = [76]
bestError = 1000.0
bestLayerSize = 0
# for i in range(100):
model = NeuralNet([76])
t = time.time()
model.fit(Xtrain, ytrain)
print("Fitting took %d seconds" % (time.time()-t))

# Compute test error
yhat = model.predict(Xval)
testError = np.mean(yhat != yval)
diff = np.mean(np.abs(yhat - yval))
print(diff)
# print(i)
print("Test error = ", testError)
# if (diff < bestError):
#     bestError = diff
#     bestLayerSize = i

print(bestError)
print(bestLayerSize)