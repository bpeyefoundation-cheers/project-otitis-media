import numpy as np
import torch
from torch.nn import Linear
from torch.nn.functional import relu
from torch import nn


class OtitisMediaClassifier(torch.nn.Module):
    """subclasssing torch.nn.Module gives us automatic gradient calculation

    Args:
        torch (_type_): _description_
    """
    def __init__(self,img_size:int, num_channels: int , num_labels:int):
        super().__init__()
        #define fully connected layer
        self.fc1=torch.nn.Linear(img_size*img_size*num_channels,512)
        self.fc2=torch.nn.Linear(512,256)
        self.fc3=torch.nn.Linear(256,128)
        self.fc4=torch.nn.Linear(128,num_labels)

    def forward(self,x:torch.Tensor):
        x=x.reshape(x.shape[0],-1)
        #forward pass with the netw
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    # 2. Modeling
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # self.linear_1 = nn.linear(5, 10)
        # self.linear_2 = nn.linear(10, 2)
        self.conv= nn.Conv2d(3, 3, kernel_size = 3 , padding= 1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_1 =  nn.Conv2d(3, 1, kernel_size = 3 , padding= 1)
        self.linear = nn.Linear(1*32 * 32 , 4)
        
    
    def forward(self , x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.conv_1(x)
        x = self.max_pool(x)
        x = self.linear(x.view(1, -1))
        return x