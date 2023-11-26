#from torch import nn
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets.image_dataset import ImageDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader









# 3. Training


if __name__ == "__main__":
    train_csv_path = r"data\train.csv"

    transforms = T.Compose([
        T.Resize((256, 256)) , T.ToTensor()
    ])
    dataset= ImageDataset(csv_path=train_csv_path, transforms=transforms)
    # print(dataset[8][0])

    data_loader = DataLoader(
        dataset, 
        batch_size = 1 , 
        shuffle = True
    )
    x, y = next(iter(data_loader))
    print(data_loader)
    model = Model()
    pred = model(x)
    # pred_label = pred.argmax().item()
    # print("pred_label")
    # BATCH_SIZE = 1
    # x = torch.rand(BATCH_SIZE, 256, 256)
    # pred = model(x)
    # print("the output is", pred)