import torch
from torch.utils.data import Dataset
from torch import nn
from utils.io import read_as_csv
import os
import numpy as np
from utils.preprocessing import image_transforms, label_transforms, label_to_index
from os.path import join
from PIL import Image

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

# 1. Prepare Datset

class ImageDataset(Dataset):
    def __init__(self , csv_path , transforms = None):
        images, labels = read_as_csv(csv_path)
        # print(images, labels)
        self.images = images
        self.labels = labels
        self.transforms = transforms
        # print(csv_path)

    def __len__(self):
        return len(self.images)
    
    def __str__(self):
        return f"<ImageDataset with {self.__len__()} samples>"

    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.labels[index]
        image_path = join("data", "middle-ear-dataset", label_name, image_name)
        image = Image.open(image_path).convert("RGB")
        label = label_to_index(label_name)

        if self.transforms:
            image = self.transforms(image)
        return image, label
