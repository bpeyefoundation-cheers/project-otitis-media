import torch 
from torch import nn 
from torch.nn import ReLU


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),padding=0)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84, 10)
        
        
    def forward(self, x):
        x = self.conv(x)
        nn.ReLU()
        x = self.max_pool(x)
        
        x= self.conv1(x)
        nn.ReLU()
        x= self.max_pool(x)
        
        x = self.conv3(x)
        x= torch.flatten(x, start_dim=1)
        
        x = self.fc1(x)
        x=self.fc2(x)
        
        
        
        return x
            
            
    
if __name__ == '__main__':
    sample_input = torch.randn(1,1, 32, 32)
    model = Model()
    output = model(sample_input)
    assert output.shape == (1,10), f'got {output.shape}'   
    print(output.shape)