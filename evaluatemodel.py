from datasets.image_dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T   
from models.customNN import FirstNeural
    
test_csv_path =r"data\test.csv"
BATCH_SIZE = 5   

model= FirstNeural(img_size= 256, num_channels = 3,  num_labels=4)

transforms= T.Compose([
    T.Resize((256, 256)), 
    T.ToTensor(),
])
train_dataset =ImageDataset( csv_path= test_csv_path , transforms= transforms)

train_data_loader= DataLoader(
    train_dataset, 
    batch_size = BATCH_SIZE,
    shuffle = True
)

for image , label in train_data_loader:
    
    model_output = model(image)
    