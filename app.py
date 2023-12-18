import torch
import requests
from PIL import Image
from torchvision import transforms as T
import gradio as gr
from models.customNN import FirstNeural

# model = torch.load(r'artifacts\run-2023-11-29-15-18-18\ckpt-FirstNeural-val=0.696-epoch=1')
# model.eval()
model_path = r"artifacts\run-2023-11-29-15-18-18\ckpt-FirstNeural-val=0.696-epoch=1"
checkpoint = torch.load(model_path)

model_state_dict  = checkpoint['model_state_dict']

model = FirstNeural(img_size= 256, num_channels=3, num_labels=4)
model.load_state_dict(model_state_dict)
model.eval()
# print(checkpoint)


labels = ['aom', 'csom', 'myringosclerosis', 'normal']

transforms= T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor()])
  
def predict(inp):
  inp = transforms(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(prediction.shape[0])}
  return confidences



gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=4),
             examples=[r"C:\Users\Dell\Downloads\normal.jpg"]).launch()