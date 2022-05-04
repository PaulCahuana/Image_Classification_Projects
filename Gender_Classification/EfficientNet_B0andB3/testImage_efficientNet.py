from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim

PATH = "resultado/model/efficientnet-b0.pth"

# Load
model = torch.load(PATH)
model.eval()

# Create the preprocessing transformation here
transform = transforms.ToTensor()

def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

from PIL import Image
# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('images/tijuana_aer.jpeg')).unsqueeze(0)
#img = tfms(Image.open('dataset_genero/test/m/61d8cf0ad0dd825c2753b614.jpg')).unsqueeze(0)
img= img.cuda()
# Load ImageNet class names
labels_map = ["f","m"]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print(torch.softmax(outputs, dim=1))
print('-----')

for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))