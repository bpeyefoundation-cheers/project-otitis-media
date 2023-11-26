#1.prepare datasets
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torchvision import transforms as T
from datasets.image_datasets import ImageDataset
from torch.utils.data import DataLoader

#2.modellings
class Model(nn.Module):
    def __init__(self):
           super().__init__()
           
           self.conv=nn.Conv2d(3,3,kernel_size=3,padding=1)
           self.max_pool=nn.MaxPool2d(kernel_size=2)
           self.conv_1=nn.Conv2d(3,1,kernel_size=3,padding=1)
           self.linear=nn.Linear(1*32*32,4)

    def forward(self,x):
        x=self.conv(x)
        x=self.max_pool(x)
        x=self.conv(x)
        x=self.max_pool(x)
        x=self.conv_1(x)
        x=self.max_pool(x)
        x=self.linear(x.view(1,-1))
        return x





      
            
        
#training

if __name__=="__main__":
    train_csv_path=r"data\train.csv"
    transforms=T.Compose([T.Resize((256,256)),T.ToTensor()])
    dataset=ImageDataset(csv_path=train_csv_path,transforms=transforms)
    data_loader=DataLoader(
         dataset,
         batch_size=1,
         shuffle=True
    )
    x,y=next(iter(data_loader))
   
    model=Model()
    pred=model(x)
    y_hat=pred.argmax().item()
    

    # BATCH_SIZE=4
    # x=torch.rand(BATCH_SIZE,3,256,256)

    
    # pred=model(x)
    print(y.item,y_hat)