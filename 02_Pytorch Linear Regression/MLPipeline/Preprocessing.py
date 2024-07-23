import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class Preprocessing:

    def __init__(self, data):
        self.data = data

    # columns to drop
    def drop(self, cols):
        col = list(cols)
        self.data.drop(col, axis=1, inplace=True)
        return self.data

    # dropping null values
    def dropna(self):
        self.data.dropna(axis=0, inplace=True)
        return self.data

    # scaling features
    def scale(self):
        num_cols = self.data.select_dtypes(
            exclude=["object"]
        ).columns.tolist()  # getting numerical columns
        scale = MinMaxScaler()
        self.data[num_cols] = scale.fit_transform(self.data[num_cols])
        return self.data

    # label encoding
    def encode(self):
        cat_cols = self.data.select_dtypes(
            include=["object"]
        ).columns.tolist()  # getting categorical columns
        le = LabelEncoder()
        self.data[cat_cols] = self.data[cat_cols].apply(le.fit_transform)
        return self.data

    # splitting data.
    def split_data(self, target_col):
        X = self.data.drop(target_col, axis=1)
        Y = self.data[target_col]

        # split a dataset into train and test sets
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.25, random_state=42
        )
        return X_train, X_test, y_train, y_test

    # converting data into tensor
    def convert_to_tensor(self, X_train, y_train, X_test, y_test):
        X_train = torch.from_numpy(X_train.values.astype(np.float32))
        X_test = torch.from_numpy(X_test.values.astype(np.float32))
        y_train = torch.from_numpy(y_train.values.astype(np.float32))
        y_test = torch.from_numpy(y_test.values.astype(np.float32))
        # #### Making output vector Y as a column vector for matrix multiplications
        y_train = y_train.view(y_train.shape[0], -1)
        y_test = y_test.view(y_test.shape[0], -1)
        n_features = X_train.shape[1]
        return n_features, X_train, y_train, X_test, y_test
