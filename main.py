#1.prepare datasets
#training
import os
from datetime import datetime
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.image_datasets import ImageDataset
from models.CustomNN import Model, OtitisMediaClassifier
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

if __name__=="__main__":
    best_val_accurcy=0
    SEED=42
    torch.manual_seed(SEED)
    BATCH_SIZE=8
    #create folder
    #get the current date and time
    dt=datetime.now()
    #format the datime with custom separators
    f_dt=dt.strftime("%Y-%m-%d-%H-%M-%S")
    folder_name=f"run-{f_dt}"


    os.mkdir(f"artifacts/{folder_name}")
    print(f"folder name:{folder_name}")

    #create tensorboard writer
    writer=SummaryWriter(log_dir=f"artifacts/{folder_name}/tensorboard_logs")
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
    #print model name
    
    #print(model(x))
#     model=Model()
#     pred=model(x)
#     # y_hat=pred.argmax().item()
    


    #training
    LR=0.001
    EPOCHS=10
    criterion=nn.NLLLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    epochwise_train_loss=[]
    epochwise_val_loss=[]
    epochwise_val_accuracy=[]
    epochwise_train_accuracy=[]

    for epoch in range(EPOCHS):
        
        train_running_loss=0
        val_running_loss=0
        train_running_accuracy=0
        val_running_accuracy=0

        
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
            # calculate train_accuracy
            preds=torch.argmax(model_out,dim=1)
            accuracy=(preds==labels).float().mean()
            train_running_accuracy+=accuracy.item()
              
            
          #validation
        model.eval()
        

        for images,labels in val_data_loader:
          
          
          model_out=model(images)
          model_out=F.log_softmax(model_out,dim=1)

          loss=criterion(model_out,labels)
          val_running_loss+=loss.item()
          
          
          # calculate val_accuracy
          preds=torch.argmax(model_out,dim=1)
          accuracy=(preds==labels).float().mean()
          val_running_accuracy+=accuracy.item()
          

          
        
          
        avg_train_loss=train_running_loss/len(train_data_loader)
        avg_val_loss=val_running_loss/len(val_data_loader)
        



        avg_val_running_accuracy=val_running_accuracy/len(val_data_loader)
        avg_train_running_accuracy=train_running_accuracy/len(train_data_loader)

        if avg_val_running_accuracy>best_val_accurcy:
          best_val_accurcy=avg_val_running_accuracy
          torch.save(model.state_dict(),f"artifacts/{folder_name}/best_model.pth")

        #log to tensorboard
        writer.add_scalar("Loss/Train",avg_train_loss,epoch)
        writer.add_scalar("Loss/val",avg_val_loss,epoch)
        writer.add_scalar("accuracy/train",avg_train_running_accuracy,epoch)
        writer.add_scalar("accuracy/val",avg_val_running_accuracy,epoch)

              
        
        epochwise_train_loss.append(avg_train_loss)
        epochwise_val_loss.append(avg_val_loss)

        epochwise_val_accuracy.append(avg_val_running_accuracy)
        epochwise_train_accuracy.append(avg_train_running_accuracy)
        if len(epochwise_val_accuracy)>=6:
          avg_epochwise_val_accuracy_for_first_5epoch=np.mean(epochwise_val_accuracy[-5:])
          print(avg_epochwise_val_accuracy_for_first_5epoch)
          avg_epochwise_val_accuracy_for_last_5epoch=np.mean(epochwise_val_accuracy[-6:-1])
          print(avg_epochwise_val_accuracy_for_last_5epoch)
          diff=avg_epochwise_val_accuracy_for_first_5epoch-avg_epochwise_val_accuracy_for_last_5epoch
          print(f"diff:{diff:.3f}")
          if diff<0.001:
            break




        
        print(f"Epoch {epoch} Train_Loss:{avg_train_loss:.3f} \t Val Loss:{avg_val_loss:.3f}  \t Tain Accuracy:{avg_train_running_accuracy:.2f}  \t Val Accuracy:{avg_val_running_accuracy:.2f}")
        checkpoint_name=f"artifacts/{folder_name}/ckpt-{model.__class__.__name__}-val_acc-{avg_val_running_accuracy:.2f}-epoch-{epoch}.pth"
        checkpoint={'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':avg_train_loss,
                    'val_loss':avg_val_loss,
                    'train_acc':avg_train_running_accuracy,
                    'val_acc':avg_val_running_accuracy
                  }
        best_checkpoint_name=f"artifacts/{folder_name}/ckpt-{model.__class__.__name__}-best_val_acc-{best_val_accurcy:.2f}-epoch-{epoch}.pth"
        
        #save the model
        torch.save(checkpoint,best_checkpoint_name)
    



    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,9))
    x_axis_values=np.arange(0,EPOCHS,1)
    ax1.plot(epochwise_train_accuracy,label='train accuracy')
    ax1.plot(epochwise_val_accuracy,label='val accuracy')
    ax1.set_title("train vs val accuracy")
    ax1.legend()
  

    ax2.plot(epochwise_val_loss,label='val loss')
    ax2.plot(epochwise_train_loss,label='train loss')
    ax2.set_title("train vs val loss")
    ax2.legend()
   
    plt.title("Loss vs accuracy")
    plt.show()


    
    




    
   



      
        