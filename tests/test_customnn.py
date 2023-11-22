import torch
import torch.optim as optim
from torch import nn

from models.CustomNN import OtitisMediaClassifier


NUM_LABELS=4
nn=OtitisMediaClassifier(img_size=512,num_labels=NUM_LABELS)
BATCH_SIZE=64
img=torch.zeros(size=(BATCH_SIZE,512,512))
output=nn.forward(img)
assert output.shape==(BATCH_SIZE,NUM_LABELS)


