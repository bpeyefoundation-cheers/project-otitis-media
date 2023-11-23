import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn 
# from models.customNN import FirstNeural
from utils.io import read_as_csv
from utils.pre_processing import label_to_index, image_transforms
import numpy as np


class OtitisMedia(Dataset):
    def __init__(self, csv_file):
        train_files, train_labels = read_as_csv(csv_file)
        
        self.imgs =  np.array([image_transforms(file, label) for file, label in zip(train_files, train_labels)])
        self.labels= np.array([label_to_index(label) for label in train_labels]) 
    
    def __len__(self ):
        return(len(self.labels))   


    
    def __getitem__(self, label=0):
        return(self.imgs[label], self.labels[label])
