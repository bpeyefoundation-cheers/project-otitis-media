import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import requests
from models.CustomNN import OtitisMediaClassifier  
import gradio as gr

checkpoint_path = r"artifacts\run-2023-12-18-14-53-15\ckpt-OtitisMediaClassifier-best_val_acc-0.85-epoch-7.pth"
checkpoint = torch.load(checkpoint_path)
transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),])
# Instantiate the model
model = OtitisMediaClassifier(img_size=256, num_channels=3, num_labels=4)

# Load the weights into the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
labels= ['aom', 'csom', 'myringosclerosis','Normal']
# Define the prediction function for Gradio
def predict(inp):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        model_out = model(inp)
        model_out = F.softmax(model_out, dim=1)
        confidences = {labels[i]: float(model_out[0][i]) for i in range(4)}  
    return confidences


# Set up Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.labels(num_top_classes=4), 
    examples=[r"data\middle-ear-dataset\Normal\n2.jpg", "data\middle-ear-dataset\myringosclerosis\mi1.jpg"]).launch()
