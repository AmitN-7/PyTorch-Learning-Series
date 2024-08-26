
# Commented out IPython magic to ensure Python compatibility.
#Importing Necessary files to read Images
import numpy as np

# Implementing a CNN in PyTorch
# importing necessary libraries

import torch
import torch.utils.data as Data
from torch import Tensor
from MLPipeline.CNNNet import CNNNet
from MLPipeline.Train import TrainModel
from MLPipeline.CreateDataset import CreateDataset

## Printing random images from the dataset
#Directory of input folder
ROOT_DIR = "Input/"

Training_folder=ROOT_DIR+"Data/Training_data"

# Setting the Image dimension and source folder for loading the dataset

IMG_WIDTH = 200 #image width
IMG_HEIGHT = 200 #image height

Train_folder = ROOT_DIR + 'Data/Training_data' #train data folder
Test_folder = ROOT_DIR + 'Data/Testing_Data' #test data folder

print("Loading Training Data")
# extract the image array and class name for training data
Train_img_data, train_class_name = CreateDataset().create_dataset(Train_folder, IMG_WIDTH, IMG_HEIGHT)
print("Training Data Loaded")

print("Loading Testing Data")
# extract the image array and class name for testing data
Test_img_data, test_class_name = CreateDataset().create_dataset(Test_folder, IMG_WIDTH, IMG_HEIGHT)
print("Testing Data Loaded")

torch_dataset_train = Data.TensorDataset(Tensor(np.array(Train_img_data)), Tensor(np.array(train_class_name)))
torch_dataset_test = Data.TensorDataset(Tensor(np.array(Test_img_data)), Tensor(np.array(test_class_name)))

# defining trainloader and testloader
trainloader = torch.utils.data.DataLoader(torch_dataset_train, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=8, shuffle=True)


#define the optimizer and loss function 
# defining the model
model = CNNNet()

TrainModel(model, ROOT_DIR, trainloader, testloader) #model training 
