#1.prepare datasets
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torchvision import transforms as T
from datasets.image_datasets import ImageDataset
from torch.utils.data import DataLoader
from models.CustomNN import Model,OtitisMediaClassifier
import torch.nn.functional as F
  
        
#training

if __name__=="__main__":
    BATCH_SIZE=16
    #prepare dataset
    train_csv_path=r"data\train.csv"
    val_csv_path=r"data\test.csv"
    transforms=T.Compose([T.Resize((256,256)),T.ToTensor()])
    train_dataset=ImageDataset(csv_path=train_csv_path,transforms=transforms)
    train_data_loader=DataLoader(
         train_dataset,
         batch_size=BATCH_SIZE,
         shuffle=True
    )
    val_dataset=ImageDataset(csv_path=val_csv_path,transforms=transforms)
    val_data_loader=DataLoader(
         val_dataset,
         batch_size=BATCH_SIZE,
         shuffle=True
    )
    #print(next(iter(val_data_loader)))

    #prepare model
    model=OtitisMediaClassifier(img_size=256,num_channels=3,num_labels=4)
    #print(model(x))
#     model=Model()
#     pred=model(x)
#     # y_hat=pred.argmax().item()
    


    #training
    LR=0.01
    EPOCHS=50
    criterion=nn.NLLLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    for epoch in range(EPOCHS):
        train_running_loss=0
        val_running_loss=0
        #traing
        model.train()


        for images,labels in train_data_loader:
            
            optimizer.zero_grad()
            model_out=model(images)
            model_out=F.log_softmax(model_out,dim=1)

            loss=criterion(model_out,labels)
            train_running_loss+=loss.item()
            loss.backward()
            optimizer.step()
            
          #validation
        model.eval()
        for images,labels in val_data_loader:
          
          # optimizer.zero_grad()
          model_out=model(images)
          model_out=F.log_softmax(model_out,dim=1)

          loss=criterion(model_out,labels)
          val_running_loss+=loss.item()
          loss.backward()
          optimizer.step()
          # calculate accuracy
          values,indices=torch.max(model_out,dim=1)
          indices==labels
        
          
        avg_train_loss=train_running_loss/len(train_data_loader)
        avg_val_loss=val_running_loss/len(val_data_loader)

        print(f"Epoch {epoch} Train_Loss:{avg_train_loss:.3f} \t Val Loss:{avg_val_loss:.3f}")
     