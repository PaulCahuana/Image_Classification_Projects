import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import json
import argparse
import pyvision.models as models
import os
# Paths for image directory and model
IMDIR="images"
MODEL='checkpoints/025x-FDMobileNet-224.pth.tar'

# Load the model for testing
model = torch.load(MODEL)
#model.load_state_dict(model['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
'''
print("model_name ",model['model_name'])
print("state_dict ",model['state_dict'])
print("optimizer ",model['optimizer'])
print("last_epoch ",model['last_epoch'])
print("best_prec1 ",model['best_prec1'])
'''

parser = argparse.ArgumentParser(description='PyTorch Classifier Training')
parser.add_argument('--data', dest='data_config', required=True, metavar='DATA_CONFIG', help='Dataset config file')
parser.add_argument('--model', dest='model_config', required=True, metavar='MODEL_CONFIG', help='Model config file')
parser.add_argument('--checkpoint', dest='checkpoint', required=True, metavar='CHECKPOINT_FILE', help='Checkpoint file')
parser.add_argument('--print-freq', dest='print_freq', default=10, type=int, metavar='N', help='Print frequency (default: 10)')


global args, best_prec1, last_epoch
args = parser.parse_args()
with open(args.data_config, 'r') as json_file:
    data_config = json.load(json_file)
with open(args.model_config, 'r') as json_file:
    model_config = json.load(json_file)
if not os.path.exists(args.checkpoint):
    raise RuntimeError('checkpoint `{}` does not exist.'.format(args.checkpoint))



print('==> Creating model `{}`...'.format(model_config['name']))

model = models.get_model(data_config['name'], model_config)
checkpoint = torch.load(args.checkpoint)
#print('==> Checkpoint name is `{}`.'.format(checkpoint['name']))
model.load_state_dict(checkpoint['state_dict'])
model = torch.nn.DataParallel(model).cuda()
print('==> Creating model completed.')
    
#model=model['state_dict']
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(model['state_dict'])
model.eval()

print(model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
#model to device
##### HELPER FUNCTION FOR FEATURE EXTRACTION
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


#model.embedding_block.register_forward_hook(get_features('feats'))

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
         #print(features['feats'].cpu().numpy())
         #print(outputs.shape)
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
