import torch
from torch import nn 

class LeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5), stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
            feature_map1= self.conv1(x)
            nn.ReLU()
            downsampled_feature_map1 = self.maxpool1(feature_map1)

            feature_map2 = self.conv2(downsampled_feature_map1)
            nn.ReLU()
            downsampled_feature_map2 = self.maxpool2(feature_map2)
            
            feature_map3 = self.conv3(downsampled_feature_map2)
            flattened1 = torch.flatten(feature_map3, start_dim=1)

            linear1= self.fc1(flattened1)
            nn.ReLU()
            linear1 = self.fc2(linear1)

            return linear1
        
if __name__ == "__main__":
    sample_input = torch.randn(1,1, 32,32)
    model = LeNet()
    output = model(sample_input)
    assert output.shape == (1, 10), f'got{output.shape}'   
    print(output.shape)  