import re
from sklearn.preprocessing import LabelEncoder
import nltk
import time
import torch
import string
import numpy as np
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.utils import resample


class Preprocessing:


    """# Function to Clean the Text"""

    # Defining a Function to Clean the Textual Data

    def clean_text(self, txt):

        txt = txt.lower()  # Lowering the text
        txt = re.sub(r'\W', ' ', str(txt)) # remove all special characters including apastrophie
        txt = txt.translate(str.maketrans('', '', string.punctuation)) # remove punctuations
        txt = ''.join([i for i in txt if not i.isdigit()]).strip() # remove digits ()
        txt = re.sub(r'\s+[a-zA-Z]\s+', ' ', txt)   # remove all single characters (it's -> it s then we need to remove s)
        txt = re.sub(r'\s+', ' ', txt, flags=re.I) # Substituting multiple spaces with single space
        txt = re.sub(r"(http\S+|http)", "", txt) # remove links
        txt = ' '.join([PorterStemmer().stem(word=word) for word in txt.split(" ") if word not in stopwords.words('english') ]) # stem & remove stop words
        txt = ''.join([i for i in txt if not i.isdigit()]).strip() # remove digits ()
        return txt

    def sampling(self,data):
        df_majority = data[data['score'] == 5]  # Data with class 5
        df_minority1 = data[data['score'] == 2]  # Data with class 2
        df_minority2 = data[data['score'] == 3]  # Data with class 3
        df_minority3 = data[data['score'] == 1]  # Data with class 1
        df_minority4 = data[data['score'] == 4]  # Data with class 4
        """# Upasampling the Monority class and Downsampling the Majority Class"""
        # Down Sampling Majority Class "5"
        df_majority_downsampled = resample(df_majority,
                                           replace=False,
                                           n_samples=600)
        # Upsample Minority class  "2"
        df_minority_upsampled = resample(df_minority1,
                                         replace=True,
                                         n_samples=200)
        # Upsample Minority class "3"
        df_minority_upsampled1 = resample(df_minority2,
                                          replace=True,
                                          n_samples=300)
        # Upsample Minority class "1"
        df_minority_upsampled2 = resample(df_minority3,
                                          replace=True,
                                          n_samples=225)
        # Upsample Minority class "4"
        df_minority_upsampled3 = resample(df_minority4,
                                          replace=True,
                                          n_samples=250)

        # Combine minority class with downsampled majority class
        data1 = pd.concat([df_majority_downsampled,
                           df_minority_upsampled,
                           df_minority_upsampled1,
                           df_minority_upsampled2,
                           df_minority_upsampled3])

        return data1

    def encoder(self,data1):
        le = LabelEncoder()
        Y = le.fit_transform(data1['score'])
        print(Y.shape)
        print(le.classes_)
        return le, Y







