import torch
import os
import numpy as np
from model import net, embedding
import argparse
from torchvision import datasets, transforms
from dataloader import custom_dset
from DeepFeatures import DeepFeatures
#os.chdir('..'); 
print(os.getcwd())



parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
parser.add_argument('--ckp', default=None, type=str,
                    help='path to load checkpoint')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()

BATCH_SIZE = 500
DATA_FOLDER = r'./Data/Employee'
IMGS_FOLDER = './Outputs/Employee/Images'
EMBS_FOLDER = './Outputs/Employee/Embeddings'
TB_FOLDER = './Outputs/Tensorboard'
EXPERIMENT_NAME = 'MNIST_VGG16'



def stack(tensor, times=3):
  return(torch.cat([tensor]*times, dim=0))

tfs = transforms.Compose([transforms.Resize((221,221)), 
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485], std=[0.229]),
                          stack])

means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)
transform  = transforms.Compose([
    transforms.Resize(size=(228, 228)),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dim = datasets.ImageFolder(root="/media/jc/LYTICA/Classification/Employee_clf/dataset/train", transform=transform)
dset_obj = custom_dset.Custom()
data_loader = torch.utils.data.DataLoader(dim, batch_size=400, shuffle=True, **kwargs)

embeddingNet = embedding.EmbeddingLeNet()

embeddingNet.eval() # Setup for inferencing

model_dict = None

try:
    model_dict = torch.load(args.ckp)['state_dict']
except Exception:
    model_dict = torch.load(args.ckp, map_location='cpu')['state_dict']


model_dict_mod = {}
for key, value in model_dict.items():
    new_key = '.'.join(key.split('.')[2:])
    model_dict_mod[new_key] = value
model = embeddingNet.to('cpu')
model.load_state_dict(model_dict_mod)

DF = DeepFeatures(model = embeddingNet, 
                  imgs_folder = IMGS_FOLDER, 
                  embs_folder = EMBS_FOLDER, 
                  tensorboard_folder = TB_FOLDER, 
                  experiment_name=EXPERIMENT_NAME)


batch_imgs, batch_labels = next(iter(data_loader))

DF.write_embeddings(x = batch_imgs)

DF.create_tensorboard_log()