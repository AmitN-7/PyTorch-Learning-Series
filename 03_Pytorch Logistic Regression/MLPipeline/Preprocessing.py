import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
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

    #scaling features
    def scale(self):
        num_cols = self.data.select_dtypes(exclude=['object']).columns.tolist()  # getting numerical columns
        scale = MinMaxScaler()
        self.data[num_cols] = scale.fit_transform(self.data[num_cols])
        return self.data

    # label encoding
    def encode(self):
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()  # getting categorical columns
        le = LabelEncoder()
        self.data[cat_cols] = self.data[cat_cols].apply(le.fit_transform)
        return self.data

    #applying all preprocessing
    def apply_preprocessing(self):
        encode_data=self.encode() 
        scale_data=self.scale()
        drop_na=self.dropna() 
        return  self.data

    #resampling data
    def resample(self,target,maj_cls,min_cls):
        df_majority = self.data[self.data[target] == maj_cls] #majority class
        df_minority = self.data[self.data[target] == min_cls] #minority class

        df_minority_upsampled = resample(df_minority, replace=True, n_samples=900, random_state=123) #upsampling minority class
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=900, random_state=123) #downsampling majority class

        #Data Concatation:  Concatanating the dataframe after upsampling and downsampling
        df2 = pd.concat([df_majority_downsampled, df_minority_upsampled]) #concanating 
        df2 = df2.sample(frac=1).reset_index(drop=True)
        return df2

    # splitting data.
    def split_data(self,df,target_col):
        X=df.drop(target_col,axis=1)
        Y=df[target_col]

        # split a dataset into train and test sets
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=42,stratify=Y)
        return X_train, X_test,y_train,y_test

    #converting data into tensor
    def convert_to_tensor(self,X_train,y_train,X_test,y_test):
        # #### Now we will convert all of them to the tensor as PyTorch works on Tensor
        X_train = torch.from_numpy(X_train.values.astype(np.float32))
        X_test = torch.from_numpy(X_test.values.astype(np.float32))
        y_train = torch.from_numpy(y_train.values.astype(np.float32))
        y_test = torch.from_numpy(y_test.values.astype(np.float32))
       #Making output vector Y as a column vector for matrix multiplications
        y_train = y_train.view(y_train.shape[0], -1)
        y_test = y_test.view(y_test.shape[0], -1)
        n_features = X_train.shape[1]
        return n_features,X_train,X_test,y_train,y_test

