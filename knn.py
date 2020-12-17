import numpy as np
from scipy import stats
import statistics
import utils
from ast import literal_eval

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        X, d = Xtest.shape
        distances = utils.euclidean_dist_squared(Xtest, self.X)

        y_pred = []
        for i in range(X):  # number of samples
            sorted_dist = np.argsort(distances[i])  # argsort returns the index that would sort the array
            nearest = sorted_dist[:self.k]
            labels = []
            for j in nearest:
                label = self.y[j]
                label = label[0]

                # y labels needed to be changed from string to actual array.
                if label[1] == ' ':
                    label = label.replace(' ', '', 1) 
                label = label.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ', ')
                processed_list = literal_eval(label) #is a string!!
                arr = np.array(processed_list)

                if (len(arr)) < 60:
                    arr = np.pad(arr, (0, 60-len(arr)), 'constant', constant_values=(0,0))
                labels.append(arr)

            labels = np.mean(labels, axis=0)
            y_pred.append(labels)

        return y_pred