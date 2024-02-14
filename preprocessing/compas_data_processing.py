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
    print("Preprocessing the COMPAS Dataset...\n")
    dataset = 'compas'
    # Load .csv file
    path = 'preprocessing/compas/compas-scores-two-years.csv'

    # Choose the features as prediction features. We chose the ones provided in the paper (Table 1)
    col_list = ["sex", "juv_fel_count", "priors_count", "race", "age_cat", "juv_misd_count", "c_charge_degree", "juv_other_count", "two_year_recid"]

    data = pd.read_csv(path, header=0, usecols=col_list)
    
    for i in range(1):
        suffled=copy.deepcopy(data)
        suffled = suffled.sample(frac=1,random_state=i).reset_index(drop=True)

        # Map categorical/qualitative attributes to numerical ones (Label Encoding)
        suffled["sex"] = suffled["sex"].map({'Male':0, 'Female':1})
        suffled["race"] = suffled["race"].map({'African-American':0, 'Asian':1, 'Caucasian':2, 'Hispanic':3, 'Native American':4, 'Other':5})
        # suffled["race"] = suffled["race"].map({'African-American':0, 'Asian':1, 'Caucasian':1, 'Hispanic':1, 'Native American':1, 'Other':1})
        suffled["age_cat"] = suffled["age_cat"].map({'Less than 25':0, '25 - 45':1, 'Greater than 45':2})
        suffled["c_charge_degree"] = suffled["c_charge_degree"].map({'M':0, 'F':1})

        # Create advantaged and disadvantaged groups
        group_label = suffled["sex"].to_numpy()

        # (Here, differently from the other datasets, there's no need to move the sensitive features 
        #  as it is already positioned at index 0)

        # Split to suffled points and ground truths
        X = suffled.iloc[:, :-1].values
        Y = suffled.iloc[:, -1].values
        
        # Standardize suffled column-wise
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
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        # Make a .npz file for the training and test datasets
        np.savez_compressed(f'dataset/{dataset}_data.npz', X_train=X_train, X_val=X_val, X_test=X_test, Y_train=Y_train, Y_val=Y_val, Y_test=Y_test)
        # Make a .npz file for the groups
        np.savez_compressed(f'dataset/{dataset}_group_label.npz', group_label=group_label)
        print("===============================================")