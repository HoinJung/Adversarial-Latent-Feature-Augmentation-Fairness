
import numpy as np
import scipy.sparse as sparse


def preprocess_data(dataset): 
    
    f = np.load(f'dataset/{dataset}_data.npz')
    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    X_val = f['X_val']
    Y_val = f['Y_val'].reshape(-1)
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)
    sensitives = X_train[:,0]
    
    if sparse.issparse(X_train):
        X_train = X_train.toarray()
    if sparse.issparse(X_val):
        X_val = X_val.toarray()
    if sparse.issparse(X_test):
        X_test = X_test.toarray()
    
    f_sense = np.load(f'dataset/{dataset}_group_label.npz')
    group_label = f_sense['group_label']
    
    A_train, A_val, A_test = group_label[X_test.shape[0]:-X_test.shape[0]:1], group_label[:X_test.shape[0]], group_label[-X_test.shape[0]::1]
    Y_train = np.expand_dims(Y_train, axis=1)
    Y_val = np.expand_dims(Y_val, axis=1)
    Y_test = np.expand_dims(Y_test, axis=1)
    
    return X_train, X_val, X_test, Y_train, Y_val,  Y_test, A_train, A_val, A_test, sensitives

