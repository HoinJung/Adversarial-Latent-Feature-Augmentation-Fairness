
import numpy as np
import scipy.sparse as sparse
from torch.utils.data import Dataset
import torch
        
class CustomDataset(Dataset):
    def __init__(self, data, target, sensitive_attribute):
        self.data = data
        self.target = target
        self.sensitive_attribute = sensitive_attribute.astype(int)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.float32)
        a = torch.tensor(self.sensitive_attribute[index], dtype=torch.float32)
        return x, y, a



def preprocess_data(dataset): 

    f = np.load(f'dataset/{dataset}_data.npz')
    X_train = f['X_train']
    Y_train = f['Y_train'].reshape(-1)
    X_val = f['X_val']
    Y_val = f['Y_val'].reshape(-1)
    X_test = f['X_test']
    Y_test = f['Y_test'].reshape(-1)

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

    
    return X_train, X_val, X_test, Y_train, Y_val,  Y_test, A_train, A_val, A_test

