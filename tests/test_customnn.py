import torch
import torch.optim as optim
from torch import nn

from models.CustomNN import OtitisMediaClassifier
from train import X_train, Y_train

NUM_LABELS=4
nn=OtitisMediaClassifier(img_size=512,num_labels=NUM_LABELS)
BATCH_SIZE=64
img=torch.zeros(size=(BATCH_SIZE,512,512))
output=nn.forward(img)
assert output.shape==(BATCH_SIZE,NUM_LABELS)


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
