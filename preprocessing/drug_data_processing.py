import numpy as np
import pandas as pd
import os
import sklearn.preprocessing as sk
import argparse
import copy

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    print("===============================================")
    print("Preprocessing the DRUG Consumption Dataset...\n")
    dataset = 'drug'
    # Load .csv file
    path = 'preprocessing/drug/drug_consumption.data'

    # Choose the features as prediction features. We chose the ones provided in the paper (Table 1):
    # ["ID", "age", "gender", "education", "country", "ethnicity", "nscore", "escore", "oscore", "ascore", "cscore", "impulsive", "ss", "coke"]
    data = pd.read_csv(path, header=None, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,20])
    #https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified
    # 16 benzos 23 Heroin 24 Mushrooms

    # Group classes to transform problem in binary classification: "CL0" (Never Used) forms class 1 (Non-user)
    # All the others ("CL1", "CL2", "CL3", "CL4", "CL5", "CL6") form class 0 (User)
    # (although this may be counterintuitive, a study of the dataset provided by the authors seems to reveal  
    #  that drug users have label 0 and non-users have label 1)

    # data[20] = data[20].map({'CL0':1, 'CL1':0, 'CL2':0, 'CL3':0, 'CL4':0, 'CL5':0, 'CL6':0,0:0,1:1})
    data[20] = data[20].map({'CL0':1, 'CL1':0, 'CL2':0, 'CL3':0, 'CL4':0, 'CL5':0, 'CL6':0})

    # Force int type to labels
    data[20] = data[20].astype(int)
    for i in range(1):
        shuffled=copy.deepcopy(data)
        shuffled = shuffled.sample(frac=1,random_state=i).reset_index(drop=True)

        # Create advantaged and disadvantaged groups
        group_label = shuffled[2].to_numpy()
        group_label = np.where(group_label<0.2,0,1)
        # Map -0.4826 (Male) to 0
        # Map 0.48246 (Female) to 1
        # group_label = np.where(data[5]==-0.31685,1,0)

        # Split to data points and ground truths
        X_unordered = shuffled.iloc[:, :-1].values

        # Move the sensitive feature to index 0 so that it is selected by default
        sensitive_feature = X_unordered[:,2] # (gender)
        X = np.hstack((sensitive_feature[..., np.newaxis], X_unordered[:,:2], X_unordered[:,3:]))
        Y = shuffled.iloc[:, -1].values

        scaler = sk.StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(f'Shape of the datapoints:           {X_scaled.shape}')
        print(f'Shape of the corresponding labels: {Y.shape}\n')

        
        idx = round(0.8*len(X_scaled))
        
        X_train = X_scaled[:idx]
        X_test = X_scaled[idx:]
        X_val = X_train[:len(X_test)]
        X_train = X_train[len(X_test):]
        Y_train = Y[:idx]
        Y_test = Y[idx:]
        Y_val = Y_train[:len(X_test)]
        Y_train = Y_train[len(X_test):]

        print(f'X_train shape: {X_train.shape}\n')
        print(f'X_val shape: {X_val.shape}\n')
        print(f'X_test shape:  {X_test.shape}\n')
        print(f'Y_train shape: {Y_train.shape}\n')
        print(f'Y_val shape: {Y_val.shape}\n')
        print(f'Y_test shape:  {Y_test.shape}\n')

        # Create output folder if it doesn't exist
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        # Make a .npz file for the training and test datasets
        np.savez_compressed(f'dataset/{dataset}_data.npz', X_train=X_train, X_val=X_val, X_test=X_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test)
        # Make a .npz file for the groups
        np.savez_compressed(f'dataset/{dataset}_group_label.npz', group_label=group_label)
    

    print("===============================================")