import torch
from torch.nn.functional import relu
from torch.nn import Linear


class First_attemptFCN(torch.nn.Module):
    """subclasssing torch.nn.Module gives us automatic gradient calculation

    Args:
        torch (_type_): _description_
    """
    def __init__(self,img_size:int,num_labels:int):
        super().__init__()
        self.fc1=torch.nn.Linear(img_size*img_size,512)
        
        self.fc2=torch.nn.Linear(512,256)
        self.fc3=torch.nn.Linear(256,128)
        self.fc4=torch.nn.Linear(128,num_labels)

    def forward(self,x:torch.Tensor):
        x=x.reshape(x.shape[0],-1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc4(x)
        return x


        #x=x.flatten()
        

    
