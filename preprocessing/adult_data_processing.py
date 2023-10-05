import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import argparse
import copy



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    data_directory = 'preprocessing/adult/'
    print("===============================================")
    print("Preprocessing the Adult Dataset...\n")
    dataset = 'adult'
    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']


    train = pd.read_csv(data_directory+'adult.data', header=None)
    test = pd.read_csv(data_directory+'adult.test', header = None,skiprows=1)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    delete_these = ['race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other', 'sex_ Female']

    delete_these += ['native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    label = df.pop('y')
    data = pd.concat([df, label], 1)


    shuffeld=copy.deepcopy(data)
    shuffeld = shuffeld.sample(frac=1,random_state=0).reset_index(drop=True)
    group_label = shuffeld['sex_ Male'].to_numpy()
    print(f'group_label shape: {group_label.shape}\n')
    print(f'group_label: {group_label}\n')

    X_unordered = shuffeld.iloc[:, :-1].values
    sensitive_feature = X_unordered[:,-2] # (gender)
    X = np.hstack((sensitive_feature[..., np.newaxis], X_unordered[:,:-2], X_unordered[:,-1:]))
    Y = shuffeld.iloc[:, -1].values

    # Standardize suffled column-wise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f'Shape of the datapoints:           {X_scaled.shape}')
    print(f'Shape of the corresponding labels: {Y.shape}\n')

    
    idx = round(0.8*len(X_scaled))
    

    ## online validationf
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

