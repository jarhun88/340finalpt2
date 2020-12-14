import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.graphics.tsaplots import plot_pacf
# import classifier
# import utils
import os
import glob

# pre-processing X train data

ROOT_DIR = os.path.abspath(os.curdir)
csvtrainingDir = ROOT_DIR
all_X_files = glob.glob(csvtrainingDir + "\\data\\train\\X\\*.csv")

# print(all_X_files)
for file in all_X_files:
    f = pd.read_csv(file, index_col=None, header=0)
    f.drop('id0', axis=1, inplace=True)
    print(f)



