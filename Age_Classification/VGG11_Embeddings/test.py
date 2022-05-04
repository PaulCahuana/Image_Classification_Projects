import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths for image directory and model
IMDIR=sys.argv[1]
MODEL='embedding_gender.pth'

# Load the model for testing
model = torch.load(MODEL)
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#model to device
##### HELPER FUNCTION FOR FEATURE EXTRACTION
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


model.embedding_block.register_forward_hook(get_features('feats'))

# Class labels for prediction
#class_names = ['client','employee'] #['0-15','16-24','25-34','35-44','45-54','55-64','65-100']   #f,m
class_names = ['f','m']
#class_names = ['0-15','16-24','25-34','35-44','45-54','55-64','65-100']

# Retreive 9 random images from directory
files=Path(IMDIR).resolve().glob('*.*')
files_l = list(files)
images=random.sample(files_l, 9)

# Configure plots
fig = plt.figure(figsize=(9,9))
rows,cols = 3,3

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=(224, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform prediction and plot results
with torch.no_grad():
    for num,img in enumerate(images):
         img=Image.open(img).convert('RGB')
         inputs=preprocess(img).unsqueeze(0).to(device)
         outputs = model(inputs)
         print(features['feats'].cpu().numpy())
         print(outputs.shape)
         _, preds = torch.max(outputs, 1)    
         label=class_names[preds]
         plt.subplot(rows,cols,num+1)
         plt.title("Pred: "+label)
         plt.axis('off')
         plt.imshow(img)
    plt.show()
'''
Sample run: python test.py test
'''
