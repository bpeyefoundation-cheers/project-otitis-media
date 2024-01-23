import torch
from torch import nn 

class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5))
        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2))
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5))
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2))
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5))
        self.flatten1=nn.Flatten()
        self.linear=nn.Linear(120,84)
        self.linear1=nn.Linear(84,10)
    def forward(self,x):
        feature_map1=self.conv1(x)
        downsampled_feature_map1=self.maxpool1(feature_map1)
        feature_map2=self.conv2(downsampled_feature_map1)
        downsampled_feature_map2=self.maxpool2(feature_map2)
        feature_map3=self.conv3(downsampled_feature_map2)
        flattened1=self.flatten1(feature_map3)
        linear=self.linear(flattened1)
        linear1=self.linear1(linear)
        return linear1  
sample_input=torch.randn(1,1,32,32)
model=LeNet()
output=model(sample_input)
assert output.shape==(1,10)