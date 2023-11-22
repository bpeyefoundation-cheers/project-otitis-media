import torch
from torch.utils.data import DataLoader,Dataset
from utils.io import read_as_csv
from utils.preprocessing import image_transforms, label_transforms
import os
import numpy as np

#create data set
class OtitisMediaDataset(Dataset):
    def __init__(self,csv_file):
        data_root=r"data"
        train_csv=r"train.csv"
        csv_file=os.path.join(data_root,train_csv)
    # load csv
        train_files, train_labels = read_as_csv(csv_file)
    # Apply transformations
        self.imgs = np.array(
            [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
    )
        self.labels = np.array([label_transforms(lab) for lab in train_labels])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,label=0):
        return self.imgs[label],self.labels[label]
        