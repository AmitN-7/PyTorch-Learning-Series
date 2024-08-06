#importing necessary libraries
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from torch import Tensor
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:

    def __init__(self, data):
        self.data=data

    #columns to drop
    def drop(self,cols):
        col=list(cols)
        self.data.drop(col,axis=1,inplace=True)
        return self.data
    
    #dropping null values
    def dropna(self):
        self.data.dropna(axis=0,inplace=True)
        return self.data 
    
    #scaling features
    def scale(self):
        num_cols=self.data.select_dtypes(exclude=['object']).columns.tolist() #getting numerical columns
        scale=MinMaxScaler()
        self.data[num_cols]=scale.fit_transform(self.data[num_cols])
        return self.data
    
    #label encoding
    def encode(self):
        cat_cols=self.data.select_dtypes(include=['object']).columns.tolist() #getting categorical columns
        le=LabelEncoder()
        self.data[cat_cols]=self.data[cat_cols].apply(le.fit_transform)
        return self.data

    #SMOTE
    def smote(self,target_col):
        #Handling Class Imbalance
        smote = SMOTE()
        # fit predictor and target variable
        # x_smote, y_smote = smote.fit_resample(self.data.iloc[:, 0:-1], self.data[target_col])
        x_smote, y_smote = smote.fit_resample(self.data.drop(target_col,axis=1), self.data[target_col])
        return x_smote,y_smote



    #splitting data. 
    def split_data(self,X,Y):
        # split a dataset into train and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size=0.2, random_state=42,stratify=Y)
        return X_train, X_test, y_train, y_test

    #data loader
    def data_loader(self,X_train,y_train):
        X_train = Tensor(X_train)
        y_train = Tensor(np.array(y_train))


        BATCH_SIZE = 64

        torch_dataset = Data.TensorDataset(X_train, y_train)

        loader = Data.DataLoader(
            dataset=torch_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=0,)
        return loader
