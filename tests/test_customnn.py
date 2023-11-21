from models.CustomNN import First_attemptFCN
import torch

NUM_LABELS=4
nn=First_attemptFCN(img_size=512,num_labels=NUM_LABELS)
BATCH_SIZE=4
img=torch.zeros(size=(BATCH_SIZE,512,512))
output=nn.forward(img)
assert output.shape==(4,4)