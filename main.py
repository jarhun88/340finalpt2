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
all_X_train_files = ROOT_DIR + "/data/train/X/*.csv"
all_Y_train_files = ROOT_DIR + "/data/train/y/*.csv"

all_X_val_files = ROOT_DIR + "/data/val/X/*.csv"
all_Y_val_files = ROOT_DIR + "/data/val/y/*.csv"

csv_processor = CSV_Processor()

Xtrain, ytrain = csv_processor.process_csv(all_X_train_files, all_Y_train_files)
Xval, yval = csv_processor.process_csv(all_X_val_files, all_Y_val_files)

print(Xtrain)
print(ytrain)
print(Xval)
print(yval)

# hidden_layer_sizes = [50]
# model = NeuralNet(hidden_layer_sizes)

print("made it")

# t = time.time()
# model.fit(Xtrain,ytrain)
# print("Fitting took %d seconds" % (time.time()-t))

# # Comput training error
# yhat = model.predict(X)
# trainError = np.mean(yhat != y)
# print("Training error = ", trainError)

# # Compute test error
# yhat = model.predict(Xtest)
# testError = np.mean(yhat != ytest)
# print("Test error     = ", testError)