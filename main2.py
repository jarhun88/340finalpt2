import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import argparse
import time
import os
import glob
import re
from ast import literal_eval
from neural_net import NeuralNet
from csv_processor import CSV_Processor
from knn import KNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    csv_processor = CSV_Processor(knnMode=True)

    if question == "1":

        ROOT_DIR = os.path.abspath(os.curdir)
        all_X_train_files = ROOT_DIR + "/data/train/X/*.csv"
        all_Y_train_files = ROOT_DIR + "/data/train/y/*.csv"

        all_X_val_files = ROOT_DIR + "/data/val/X/*.csv"
        all_Y_val_files = ROOT_DIR + "/data/val/y/*.csv"

        all_X_test_files = ROOT_DIR + "/data/test/X/*.csv"

        print("starting...")


        #Xtrain, ytrain = csv_processor.process_csv(all_X_train_files, all_Y_train_files)
        #Xval, yval = csv_processor.process_csv(all_X_val_files, all_Y_val_files)
        Xtest = csv_processor.process_test_csv(all_X_test_files)

        pd.DataFrame(Xtest).to_csv('Xtestprocessed.csv')
        #pd.DataFrame(Xtrain).to_csv('Xtrainprocessed.csv')
        #pd.DataFrame(ytrain).to_csv('ytrainprocessed.csv')
        #pd.DataFrame(Xval).to_csv('Xvalprocessed.csv')
        #pd.DataFrame(yval).to_csv('yvalprocessed.csv')

    if question == '2':
        dfx=pd.read_csv('Xtrainprocessed.csv', sep=',', header=0, index_col=0)
        dfy = pd.read_csv('ytrainprocessed.csv', sep=',', header=0, index_col=0)

        dfxtest = pd.read_csv('Xtestprocessed.csv', sep=',', header=0, index_col=0)
        dfyval = pd.read_csv('yvalprocessed.csv', sep=',', header=0, index_col=0)
        
        # yval cleaning:
        yval = dfyval.to_numpy()
        Xtest = dfxtest.to_numpy()
  
        X = dfx.to_numpy() 
        print(X.shape)    
        y = dfy.to_numpy()
        print(y.shape)


        for n in range(5):
            bestErr = 100000
            bestn = 0

            model = KNN(k=n)

            model.fit(X, y)

            print('predicting...')
        
            y_pred = model.predict(Xtest)

            # print(y_pred)

            test_error_abs = 0
            test_error_mse = 0
            for i in range(len(y_pred)):
                label = y[i]
                label = label[0]
                if label[1] == ' ':
                    # print('has a space', label)
                    label = label.replace(' ', '', 1) 
                label = label.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ', ')
                y_temp = literal_eval(label)
                arr = np.array(y_temp)
                if (len(arr)) < 60:
                    arr = np.pad(arr, (0, 60-len(arr)), 'constant', constant_values=(0,0))
                
                test_error_mse += np.sum((y_pred[i] - arr)**2)
                test_error_abs += np.sum(np.abs(y_pred[i] - arr))
            
            print("=====", n, "=====")
            print("KNN Absolute Value Error: %.3f" % test_error_abs)
            print("KNN Mean Squared Error: %.3f" % test_error_mse)

            if test_error_mse < bestErr:
                bestErr = test_error_mse
                bestn = n 
        print(test_error_mse)
        print(bestn)

            # saving results to a csv submission file
            # print(y_pred)
        please = np.array(y_pred)
        csv_processor.to_kaggle_csv(please)
        
        
        '''
        idcolumn = []
        locolumn = []
        # for idn in range(X.shape[0]):  # number of samples 
        
        for idn in range(20):
            data = y_pred[idn]
            for time in range(1, 30): #  number of seconds
                for x in range(2):
                    if x == 0:
                        z = 'x'
                    else: 
                        z = 'y'
                    idd = "{}_{}_{}".format(idn, z, time)
                    idcolumn.append(idd)
                    locolumn.append(data[time*2+x]) # <- this is funky         
        

        data = {'id': idcolumn,
        'location': locolumn
        }
        
        df = pd.DataFrame(data, columns= ['id', 'location'])
        df.to_csv('submission.csv', index=False, header=True)

        # print(df)
       '''

    
                    


          
        



# hidden_layer_sizes = [50]
# model = NeuralNet(hidden_layer_sizes)



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