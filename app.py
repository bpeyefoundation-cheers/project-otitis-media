import gradio as gr
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
from models.CustomNN import OtitisMediaClassifier

checkpoint_path = r"artifacts\run-2023-12-18-14-53-15\ckpt-OtitisMediaClassifier-best_val_acc-0.85-epoch-8.pth"
checkpoint = torch.load(checkpoint_path)

print(checkpoint.keys())

transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),])
# Instantiate the model
model = OtitisMediaClassifier(img_size=256, num_channels=3, num_labels=4)

# Load the weights into the model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
labels= ['aom', 'csom', 'myringosclerosis','Normal']

threshold_proba=0.5


test_csv_path = r"data\test.csv"
test_df = pd.read_csv(test_csv_path)

test_set = test_df['file'].tolist()  
true_labels = test_df['label'].tolist()
# Define the prediction function for Gradio
def predict(inp,csv_file="model_pred.csv"):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        model_out = model(inp)
        model_out = F.softmax(model_out, dim=1)
        confidences = {labels[i]: float(model_out[0][i]) for i in range(4)} 
        predictions_df = pd.DataFrame({
        "Image_Path": test_set,
        "True_Label": true_labels,
        **confidences
    })

    # Save the DataFrame to a new CSV file
    predictions_df.to_csv(csv_file, index=False)
        # diff=max(confidences.values())-list(sorted(confidences. values()))[-2]
        
        # if diff < threshold_proba:

        #     result= "Consult with a doctor"

    #
    

# Set up Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    examples=test_set[:2],  # Provide some examples from the test set for Gradio to use
).launch()