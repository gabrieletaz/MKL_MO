import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from .args import args
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from src.utils import *

# ----- Dataset -----
class SingleOmicDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        x = self.data[0][index]
        y = self.data[1][index]

        return x, y


def import_dataset(dataset=args.dataset):
    path = "/workspaces/nn_integration/data/" + dataset

    labels_train = pd.read_csv(path + "/labels_tr.csv", header=None) 
    labels_test = pd.read_csv(path + "/labels_te.csv", header=None)
    labels = pd.concat([labels_train, labels_test], ignore_index=True)
        
    # omic 1 (mRNA)
    data_1_train = pd.read_csv(path + "/1_tr.csv", header=None)
    data_1_test = pd.read_csv(path + "/1_te.csv", header=None)
    omic_1 = pd.concat([data_1_train, data_1_test], ignore_index=True)
        
    # omic 2 (DNA methylation)
    data_2_train = pd.read_csv(path + "/2_tr.csv", header=None)
    data_2_test = pd.read_csv(path + "/2_te.csv", header=None)
    omic_2 = pd.concat([data_2_train, data_2_test], ignore_index=True)

    # omic 3 (miRNA)
    data_3_train = pd.read_csv(path + "/3_tr.csv", header=None)
    data_3_test = pd.read_csv(path + "/3_te.csv", header=None)
    omic_3 = pd.concat([data_3_train, data_3_test], ignore_index=True)

    return omic_1, omic_2, omic_3, labels

def load_data(seed, kernel_par, n_principal):
    fix_random_seed(seed=seed)

    omic1, omic2, omic3, labels = import_dataset() 
    y = labels.values.astype(np.int64)
    indices = np.arange(len(y))
    
    
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y)
    y_tr = y[train_indices]
    y_te = y[test_indices]
    #print(train_indices)

    #omic1 (mRNA) dataset
    X_1 = omic1.values.astype(np.float64)
    X_1_tr = X_1[train_indices]
    K_PCA_1 = KernelPCA(n_components = n_principal, kernel='rbf', gamma=kernel_par[0], random_state=seed)
    X_1_tr = K_PCA_1.fit_transform(X_1_tr)
    X_1_te = X_1[test_indices]
    X_1_te = K_PCA_1.transform(X_1_te)

    train_1 = SingleOmicDataset(data=(X_1_tr, y_tr))
    test_1  = SingleOmicDataset(data=(X_1_te, y_te))

    #omic2 (mRNA)  dataset
    X_2 = omic2.values.astype(np.float64)
    X_2_tr = X_2[train_indices]
    K_PCA_2 = KernelPCA(n_components = n_principal, kernel='rbf', gamma=kernel_par[1], random_state=seed)
    X_2_tr = K_PCA_2.fit_transform(X_2_tr)
    X_2_te = X_2[test_indices]
    X_2_te = K_PCA_2.transform(X_2_te)

    train_2 = SingleOmicDataset(data=(X_2_tr, y_tr))
    test_2  = SingleOmicDataset(data=(X_2_te, y_te))

    #omic3 (mRNA) dataset
    X_3 = omic3.values.astype(np.float64)
    X_3_tr = X_3[train_indices]
    K_PCA_3 = KernelPCA(n_components = n_principal, kernel='rbf', gamma=kernel_par[2], random_state=seed)
    X_3_tr = K_PCA_3.fit_transform(X_3_tr)
    X_3_te = X_3[test_indices]
    X_3_te = K_PCA_3.transform(X_3_te)

    train_3 = SingleOmicDataset(data=(X_3_tr, y_tr))
    test_3  = SingleOmicDataset(data=(X_3_te, y_te))

    return [train_1, test_1], [train_2, test_2], [train_3, test_3]


if __name__ == "__main__":
    pass
