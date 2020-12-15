import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import time
import os
import glob
from neural_net import NeuralNet
from csv_processor import CSV_Processor
from sklearn.preprocessing import LabelBinarizer


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

# binarizer = LabelBinarizer()
# Ytrain = binarizer.fit_transform(ytrain)

print(Xtrain)
print(ytrain)
# print(Xval)
# print(yval)

print("made it")

hidden_layer_sizes = [50]
model = NeuralNet(hidden_layer_sizes)

t = time.time()
model.fit_SGD(Xtrain, ytrain)
print("Fitting took %d seconds" % (time.time()-t))

# Comput training error
yhat = model.predict(Xval)
trainError = np.mean(yhat != yval)
print("Training error = ", trainError)

# Compute test error
yhat = model.predict(Xval)
testError = np.mean(yhat != yval)
# print("Test error     = ", testError)

