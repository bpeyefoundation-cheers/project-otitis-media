import torch
from torch.utils.data import Dataset, Dataloaders
from torch import nn
from models.CustomNN import OtitisMediaClassifier
from utils.io import read_as_csv
import os
import numpy as np
from utils.preprocessing import image_transforms, label_transforms

IMG_SIZE = 256
NUM_CHANNELS = 3
NUM_LABELS = 4

#create model object
model_nn = OtitisMediaClassifier(img_size= IMG_SIZE, num_channels = NUM_CHANNELS, num_labels = NUM_LABELS)

