import torch
import requests
from PIL import Image
from torchvision import transforms as T
import gradio as gr
from models.CustomNN import OtitisMediaClassifier

#load the model
model_path = r"artifacts\run-2023-12-18-15-04-36\ckpt-OtitisMediaClassifier-val-acc-0.48-epoch=2"
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

def predict(inp):
  inp = transforms(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(4)}
  return confidences

gr.Interface(fn= predict,
             inputs = gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=4),
             examples=[r"data\middle-ear-dataset\aom\aom_13.tiff"]).launch()
             

