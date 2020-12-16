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

all_X_test_files = ROOT_DIR + "\\data\\test\\X\\"

sample_submission = ROOT_DIR + "\\data\\sample_submission.csv"

csv_processor = CSV_Processor()

Xtrain, ytrain = csv_processor.process_csv(all_X_train_files, all_Y_train_files)
Xtest = csv_processor.process_test_csv(all_X_test_files)

hidden_layer_sizes = [76]
model = NeuralNet(hidden_layer_sizes)
t = time.time()
model.fit(Xtrain, ytrain)
print("Fitting took %d seconds" % (time.time()-t))

# Compute test error
yhat = model.predict(Xtest)
retcsv = yhat.flatten()
retcsv2 = pd.DataFrame(retcsv, columns=["location"])

# np.savetxt("y_actual.csv", yhat, delimiter=", ")
# np.savetxt("y_expected.csv", yval, delimiter=", ")

retcsv2.to_csv("y_pred.csv", index=False)

# retcsv2 = pd.read_csv(ROOT_DIR + "\\y_pred.csv")

df = csv_processor.process_output_csv(sample_submission)
df.drop(df.columns[1], axis=1, inplace=True)
df2 = df.join(retcsv2)
df2.to_csv('y_output.csv', index=False)
