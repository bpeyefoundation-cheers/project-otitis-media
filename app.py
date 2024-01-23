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
  threshold = 0.5
  inp = transforms(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(prediction.shape[0])}
    # print(confidences)
  
    sorted_confidences = sorted(confidences.items(), key=lambda x: x[1])
    # print(sorted_confidences)
    diff = sorted_confidences[3][1] - sorted_confidences[2][1]
    if diff < threshold:
      return "unable to predict"
    
  return confidences



gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=4),
             examples=[r"C:\Users\Dell\Downloads\normal.jpg"]).launch()



























  #   # Finding top classes and their confidences
    # sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    # top_classes = sorted_confidences[:4]
        
    #     # Checking if the difference between top confidences is less than threshold
    # if len(top_classes) > 1 and (top_classes[0][1] - top_classes[1][1]) < threshold:
    #         return {'Unable to predict': 1.0}  # Return 'Unable to predict' label