import io

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.CustomNN import OtitisMediaClassifier

checkpoint_path = r"artifacts\run-2023-11-30-17-37-02\ckpt-OtitisMediaClassifier-best_val_acc-0.88-epoch-8.pth"
checkpoint = torch.load(checkpoint_path)

# print(checkpoint.keys())

transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),])
# Instantiate the model
model = OtitisMediaClassifier(img_size=256, num_channels=3, num_labels=4)

# Load the weights into the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
labels= ['aom', 'csom', 'myringosclerosis','Normal']

threshold_proba=0.5

import os

# print(os.getcwd())


test_csv_path = os.path.join(r"C:\\Users\\Dell\\Desktop\\project1\\project-otitis-media", "data", "test.csv")

test_df = pd.read_csv(test_csv_path)



test_set = test_df['file'].tolist()

true_labels = test_df['label'].tolist()



example_file_paths = []


classes_for_testing = ['aom', 'csom', 'myringosclerosis', 'Normal']
images_per_class = 2  

for class_label in classes_for_testing:
    class_folder = os.path.join(r"C:\Users\Dell\Desktop\project1\project-otitis-media\data", "middle-ear-dataset", class_label)
    class_images = os.listdir(class_folder)[:images_per_class]
    class_paths = [os.path.join(class_folder, filename) for filename in class_images]
    example_file_paths.extend(class_paths)
#Define the prediction function for Gradio
def predict(inp, csv_file="model_pred.csv"):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        model_out = model(inp)
        model_out = F.softmax(model_out, dim=1)
        confidences = {labels[i]: float(model_out[0][i]) for i in range(4)}

    return confidences


#Set up Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    examples=example_file_paths,
).launch()

