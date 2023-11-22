import numpy as np
import torch
from torch.nn import Linear
from torch.nn.functional import relu


class OtitisMediaClassifier(torch.nn.Module):
    """subclasssing torch.nn.Module gives us automatic gradient calculation

    Args:
        torch (_type_): _description_
    """
    def __init__(self,img_size:int,num_labels:int):
        super().__init__()
        #define fully connected layer
        self.fc1=torch.nn.Linear(img_size*img_size,512)
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


        #x=x.flatten()
        
# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).long()

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = OtitisMediaClassifier()
# define a loss function
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 20
train_losses= []

# Training loop
# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#         print(f'Epoch: {e+1} \t Training Loss: {running_loss / len(trainloader)}')

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        log_probs =model(images)
        loss = criterion(output, labels)
        loss.backward() 
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    

        running_loss = running_loss / len(trainloader.sampler)
        train_losses.append(running_loss)


        print(f'Epoch: {e+1} \t Training Loss: {running_loss:.6f} ')
