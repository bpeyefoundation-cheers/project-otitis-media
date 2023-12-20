import torch
import requests
from PIL import Image
from torchvision import transforms as T
import gradio as gr
from models.CustomNN import OtitisMediaClassifier
import pandas as pd

#load the model
model_path = r"artifacts\run-2023-12-18-15-04-36\ckpt-OtitisMediaClassifier-val-acc-0.79-epoch=8"
checkpoint = torch.load(model_path)

model_state_dict = checkpoint['model_state_dict']
# print(model_state_dict)
model = OtitisMediaClassifier(img_size= 256, num_channels=3, num_labels=4)
model.load_state_dict(model_state_dict)
model.eval()

# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")
labels = ['aom', 'csom', 'myringosclerosis', 'normal']

transforms= T.Compose([
  T.Resize((256, 256)),
  T.ToTensor()])

# Download human-readable labels for ImageNet.
# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")

threshold = 0.5
test_csv_path = r'data\tests.csv'
data = pd.read_csv(test_csv_path)

test_cases = data['file'].tolist()
predictions = data['label'].tolist()

# print(test_cases)

def predict(inp, csv_file= "model_pred.csv"):
  inp = transforms(inp).unsqueeze(0)
  
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    
    confidences = {labels[i]: float(prediction[i]) for i in range(4)}
    # print(confidences)
    
    sorted_confidences = sorted(confidences.items(), key=lambda x:x[1])
    # print(sorted_confidences)
    differences = sorted_confidences[3][1] - sorted_confidences[2][1]
    # print(differences)
    if differences < threshold:
        return "unable to predict"
    


  return confidences





gr.Interface(fn= predict,
             inputs = gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=4),
             examples=[r"data\middle-ear-dataset\aom\aom_13.tiff"]).launch()
             

