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



if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    BATCH_SIZE =16
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
    EPOCHS = 10
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters() , lr = LR)
    epochwise_train_losses = []
    epochwise_val_losses = []
    epochwise_val_acc = []
    epochwise_train_acc = []
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
          

        avg_train_loss = train_running_loss/ len(train_data_loader)
        avg_val_loss = val_running_loss / len(val_dataloader)
        epochwise_train_losses.append(avg_train_loss)
        epochwise_val_losses.append(avg_val_loss)
        epochwise_val_acc.append(avg_val_accuracy)
        epochwise_train_acc.append(avg_train_accuracy)

        
        print(f"Epoch {epoch} \t Train Loss : {avg_train_loss:.3f} \t Val loss : {avg_val_loss:.2f} \t Val Accuracy:{avg_val_accuracy:.2f} \t Train Accuracy:{avg_train_accuracy:.2f}") 
    

    #save the model
     
    torch.save(model.state_dict(), "artifacts/first_model.pt")
    
    torch.save(model, "atifacts/first_model_1.pt")


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