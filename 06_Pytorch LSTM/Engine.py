# -*- coding: utf-8 -*-

import re
import tensorflow as tf
import numpy as np
import nltk

nltk.download('stopwords')
from tensorflow import keras
import torch.nn.functional as F
#import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from MLPipeline.Load_Data import Load_Data
from MLPipeline.Preprocessing import Preprocessing
from MLPipeline.Tokenisation import Tokenisation
from MLPipeline.Create import Create
from MLPipeline.Lstm import LSTM
from MLPipeline.Train_Test import Train_Test

max_features = 2000
batch_size = 50
vocab_size = max_features

# lOADING THE DATA
data = Load_Data().load_data()
print(data)

preprocess = Preprocessing()

"""# Applying the Function to the Dataset"""

data['content'] = data['content'].apply(preprocess.clean_text)  # apply the function to every text in the dataset

# we can see that class "5"  is dominating in the dataset. Thus we need to Balance the Dataset.

"""# Balancing the Dataset"""

data1 = preprocess.sampling(data)

""" # Tokenisation """
word_index, X = Tokenisation().generate_token(data1)

"""# Label Encodning """
le, Y = preprocess.encoder(data1)

creating = Create()

X_train, X_test, Y_train, Y_test = creating.create_dataset(X, Y)

x_cv, y_cv, train_dl, val_dl = creating.data_loader(X_train, X_test, Y_train, Y_test)

"""# Defining the Model"""
model = LSTM(vocab_size, 128, 64)
print(model)

Train_Test().train_test(10, model, train_dl, x_cv, val_dl, Y_test)
