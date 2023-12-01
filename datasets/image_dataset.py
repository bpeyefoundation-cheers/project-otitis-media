import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# from models.customNN import FirstNeural
from utils.io import read_as_csv
from utils.pre_processing import label_to_index, image_transforms, read_image
import numpy as np
from os.path import join
from PIL import Image


class OtitisMedia(Dataset):
    def __init__(self, csv_file):
        train_files, train_labels = read_as_csv(csv_file)

        self.imgs = np.array(
            [
                image_transforms(file, label)
                for file, label in zip(train_files, train_labels)
            ]
        )
        self.labels = np.array([label_to_index(label) for label in train_labels])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, label=1):
        return (self.imgs[label], self.labels[label])


class ImageDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        images, labels = read_as_csv(csv_path)
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __str__(self):
        return f"<ImageDatset with {self.__len__()} samples>"

    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.labels[index]
        image_path = join("data", "middle-ear-dataset", label_name, image_name)
        # print(image_path)
        image = Image.open(image_path).convert("RGB")

        label = label_to_index(label_name)
        if self.transforms:
            image = self.transforms(image)

        return image, label
