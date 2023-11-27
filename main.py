#from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.image_dataset import ImageDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from models.CustomNN import Model , OtitisMediaClassifier




if __name__ == "__main__":
    BATCH_SIZE =16
    train_csv_path = r"data\train.csv"
    val_csv_path = r"data\train.csv"

    transforms = T.Compose([
        T.Resize((256, 256)) , T.ToTensor()
    ])
    train_dataset= ImageDataset(csv_path=train_csv_path, transforms=transforms)
    val_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

    val_dataset = ImageDataset(csv_path= val_csv_path, transforms=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = True)

    # print(next(iter(val_dataloader)))
    x, y = next(iter(val_data_loader))
    # print(val_data_loader)
    # model = Model()
    # pred = model(x)

# 3. model
    model = OtitisMediaClassifier(img_size=256 , num_labels=4, num_channels = 3)


    #train
    LR = 0.000001
    EPOCHS = 10
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = LR)
    
    for epoch in range(EPOCHS):
        train_running_loss = 0
        val_running_loss = 0
        for images , labels in val_data_loader:
            optimizer.zero_grad()
            model_out  = model(images)
            model_out = F.log_softmax(model_out, dim = 1)
            loss  = criterion(model_out , labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        #validation
        model.eval()  #change into validation mode
        for images , labels in val_data_loader:
            model_out  = model(images)
            model_out = F.log_softmax(model_out, dim = 1)
            loss  = criterion(model_out , labels)
            val_running_loss += loss.item()
          

        avg_train_loss = train_running_loss/ len(val_data_loader)
        avg_val_loss = val_running_loss / len(val_dataloader)
        print(f"Epoch {epoch} Train Loss : {avg_train_loss:.3f} val loss : {avg_val_loss:.2f}") 

        
        #CALCULATE ACCURACY

    values , indices =torch.

    
    # print(model(x))
    # pred_label = pred.argmax().item()
    # print("pred_label")
    # BATCH_SIZE = 1
    # x = torch.rand(BATCH_SIZE, 256, 256)
    # pred = model(x)
    # print("the output is", pred)