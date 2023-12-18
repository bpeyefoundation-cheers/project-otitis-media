#from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets.image_dataset import ImageDataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from models.CustomNN import Model , OtitisMediaClassifier
import matplotlib.pyplot as plt
from uuid import uuid4
import os
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    #create folder
    
    #get the current date and time
    dt = datetime.now()

    #format the date and time with custom seperators
    format_dt = dt.strftime("%Y-%m-%d-%H-%M-%S")
    folder_name = f"run-{format_dt}"
    os.mkdir(f"artifacts/{folder_name}")
    print(f"Folder name: {folder_name}")

    #create a tensorboard writer

    writer = SummaryWriter(log_dir = f"artifacts/{folder_name}/tensorboard_logs")

    SEED = 42
    torch.manual_seed(SEED)
    BATCH_SIZE =8
    train_csv_path = r"data\train.csv"
    val_csv_path = r"data\tests.csv"

    transforms = T.Compose([
        T.Resize((256, 256)) , T.ToTensor()
    ])
    train_dataset= ImageDataset(csv_path=train_csv_path, transforms=transforms)
    train_data_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

    val_dataset = ImageDataset(csv_path= val_csv_path, transforms=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle = True)



# 3. model
    model = OtitisMediaClassifier(img_size=256 , num_labels=4, num_channels = 3)


    #train
    LR = 0.000001
    EPOCHS = 12
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = LR)
    epochwise_train_losses = []
    epochwise_val_losses = []
    epochwise_val_acc = []
    epochwise_train_acc = []
    best_acc = 0
    for epoch in range(EPOCHS):
        train_running_loss = 0
        val_running_loss = 0

        train_running_accuracy = 0
        val_running_accuracy = 0
        for images , labels in train_data_loader:
            optimizer.zero_grad()
            model_out  = model(images)
            model_out = F.log_softmax(model_out, dim = 1)
            loss  = criterion(model_out , labels)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()



            
        
        #train
        model.eval()  #change into validation mode
        for images , labels in train_data_loader:

            model_out  = model(images)
            model_out = F.log_softmax(model_out, dim = 1)
            
            loss  = criterion(model_out , labels)
            val_running_loss += loss.item()

            #accuracy

            preds = torch.argmax(model_out, dim =1)
            acc = (preds == labels).float().mean()
            train_running_accuracy += acc.item()

        avg_train_accuracy = train_running_accuracy / len(train_data_loader)



        #validation

        for images , labels in val_dataloader:

            model_out  = model(images)
            model_out = F.log_softmax(model_out, dim = 1)
            
            loss  = criterion(model_out , labels)
            val_running_loss += loss.item()

             #accuracy

            preds = torch.argmax(model_out, dim =1)
            acc = (preds == labels).float().mean()
            val_running_accuracy += acc.item()

        avg_val_accuracy = val_running_accuracy / len(val_dataloader)
        
        if best_acc<avg_val_accuracy:
            best_acc = avg_val_accuracy
            torch.save(model.state_dict(), f"artifacts/{folder_name}/best_model.pth")
        


          

        avg_train_loss = train_running_loss/ len(train_data_loader)
        avg_val_loss = val_running_loss / len(val_dataloader)
        epochwise_train_losses.append(avg_train_loss)
        epochwise_val_losses.append(avg_val_loss)
        epochwise_val_acc.append(avg_val_accuracy)
        epochwise_train_acc.append(avg_train_accuracy)
        
        #log to tensrboard
        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("accuracy/train", avg_train_accuracy, epoch)
        writer.add_scalar("accuracy/val", avg_val_accuracy, epoch)

        
        #early stopping

        if len(epochwise_val_acc) >= 6 :
            last_5_elements1 = epochwise_val_acc[-5:]
            acc_avg1 = np.mean(last_5_elements1)
            print("average of last 5 elements : ", acc_avg1)
            last_5_elements2 = epochwise_val_acc[-6 :-1]
            acc_avg2 = np.mean(last_5_elements2)
            print("average of 2nd last 5 elements : ",acc_avg2)

            difference = acc_avg1 - acc_avg2
            print("differences : ", difference)
            if difference < 0.001 :
                break

        

            

            

        
        print(f"Epoch {epoch} \t Train Loss : {avg_train_loss:.3f} \t Val loss : {avg_val_loss:.2f} \t Val Accuracy:{avg_val_accuracy:.2f} \t Train Accuracy:{avg_train_accuracy:.2f}") 


        # torch.save(model.state_dict(), f"artifacts/{epoch}_model.pt")
        checkpoint_name = f"artifacts/{folder_name}/ckpt-{model.__class__.__name__}-val-acc-{avg_val_accuracy:.2f}-epoch={epoch}"
        checkpoint = {
            "epoch" : epoch,
            "model_state_dict" :model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "train_loss" : avg_train_loss,
            "val_loss" : avg_val_loss,
            "train_acc" : avg_train_accuracy,
            "val_acc" : avg_val_accuracy

        }
        torch.save(checkpoint, checkpoint_name)
        
        

        
    # print("Maximum Accuracy", best_acc)
    

    #save the model
     
    # torch.save(model.state_dict(), "artifacts/first_model.pt")
    
    # torch.save(model, "artifacts/first_model_1.pt")


    #     #plot the losses
    # plt.plot(epochwise_train_losses, label="train loss")
    # plt.plot(epochwise_val_losses, label = "validation loss")
    # plt.plot(epochwise_val_acc, label = "validation accuracy")
    # plt.plot(epochwise_train_acc, label = "train accuracy")
    
    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(16,9))
    ax1.plot(epochwise_train_losses, label="Train loss")
    ax1.plot(epochwise_val_losses, label="Validation loss")
    ax1.set_title("Train vs Validation Loss")
    ax1.legend()
    # ax1.imshow()


    ax2.plot(epochwise_train_acc, label ="Train accuracy")
    ax2.plot(epochwise_val_acc, label ="Validation accuracy")
    ax2.set_title("Train vs Validation Accuracy")
    ax2.legend()
    # ax2.imshow()

    plt.show()





    # plt.plot()
    # plt.legend()
    # plt.show()

        
        #CALCULATE ACCURACY

    # values , indices =torch.``

    
    # print(model(x))
    # pred_label = pred.argmax().item()
    # print("pred_label")
    # BATCH_SIZE = 1
    # x = torch.rand(BATCH_SIZE, 256, 256)
    # pred = model(x)
    # print("the output is", pred)