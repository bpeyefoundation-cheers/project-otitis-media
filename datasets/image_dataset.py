import torch
from torch.utils.data import Dataset
from torch import nn
from utils.io import read_as_csv
import os
import numpy as np
from utils.preprocessing import image_transforms, label_transforms

#create dataset
class OtitisMediaClassifier(Dataset):
    def __init__(self, csv_file):
        # load csv
        # data_root = "data\middle-ear-dataset"
        # train_csv = "data\train.csv"
        # train_path = os.path.join(data_root , train_csv)
        train_files , train_labels = read_as_csv(csv_file)
       

        #apply the image tranform function to train and test file
        self.imgs = np.array([image_transforms(file_name , label) for file_name , label in zip(train_files, train_labels)])
        self.labels = np.array([label_transforms(label) for label in train_labels])
    
    def __len__(self):
        return(len(self.labels))
    
    def __getitem__(self, label= 0):

        return(self.imgs[label], self.labels[label])
