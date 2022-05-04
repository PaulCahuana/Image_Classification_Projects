import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import multiprocessing
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch import nn

# Paths for image directory and model
IMDIR=sys.argv[1]
MODEL='embedding_gender.pth'


# Load the model for testing
model = torch.load(MODEL)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

model.embedding_block.register_forward_hook(get_features('feats'))

preprocess=transforms.Compose([
        transforms.Resize(size=(224, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# placeholders
PREDS = []
FEATS = []

# placeholder for batch features
features = {}


#charge file of images
files=Path(IMDIR).resolve().glob('*.*')
files_l = list(files)
images=files_l #random.sample(files_l, 9)

# loop through batches
for idx, img in enumerate(images):
    img=Image.open(img).convert('RGB')
    inputs=preprocess(img).unsqueeze(0).to(device)
    # move to device
    inputs = inputs.to(device)  
    print(inputs)
    # forward pass [with feature extraction]
    preds = model(inputs)
    
    # add feats and preds to lists
    PREDS.append(preds.detach().cpu().numpy())
    FEATS.append(features['feats'].cpu().numpy())


PREDS = np.concatenate(PREDS)
FEATS = np.concatenate(FEATS)


m = nn.Sigmoid()
PREDS=torch.from_numpy(PREDS)
PREDS = m(PREDS)


print('- preds shape:', PREDS.shape)
print('- feats shape:', FEATS.shape)

print('- preds :', PREDS)
print('- feats :', FEATS)














