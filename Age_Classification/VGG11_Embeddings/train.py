import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import time, os, copy, argparse
import multiprocessing
from torchsummary import summary
from matplotlib import pyplot as plt

# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--data_path", required=True, help="dataset path")
ap.add_argument("--type", required=True, help="Dataset mode: gender/age/employee/embedding")
ap.add_argument("--mode", required=True, help="Training mode: finetue/transfer/scratch")
ap.add_argument("--save_dir", help="Save checkpoint directory")
ap.add_argument("--batch_size", default=16, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
ap.add_argument("--epochs", default=26, type=int, metavar="N", help="number of total epochs to run")
args= vars(ap.parse_args())

# Set training mode
train_mode = args["mode"] #scratch

# Set the train and validation directory paths
train_directory = args["data_path"]+'/train'
valid_directory = args["data_path"]+'/val'
# Set the model save path
PATH="embedding_gender.pth" 

# Batch size
bs = args["batch_size"]
# Number of epochs
num_epochs = args["epochs"]
# Number of classes
num_classes = 7
# Number of workers
num_cpu = multiprocessing.cpu_count()
# dataset type
type_mode = args["type"]


# Applying transforms to the data
image_transforms = { 
    'train': transforms.Compose([
        
        transforms.Resize(size=(224, 128)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(224, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
 
# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=True)
}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if train_mode=='finetune':
    # Load a pretrained model - Resnet18
    print("\nLoading resnet18 for finetuning ...\n")
    model_ft = models.resnet18(pretrained=True)

    # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes )

elif train_mode=='scratch':
    # Load a custom model - VGG11
    if type_mode == "gender":
        print("\nLoading VGG11 for training from scratch ...\n")
        model_ft = MyVGG11(in_ch=3,num_classes=2)
    elif type_mode == "age":
        print("\nLoading VGG11 for training from scratch ...\n")
        model_ft = ageVGG11(in_ch=3,num_classes=len(class_names))
    elif type_mode == "employee":
        print("\nLoading VGG11 for training from scratch ...\n")
        model_ft = MyVGG11(in_ch=3,num_classes=2)
    elif type_mode == "embedding":
        print("\nLoading embeddingVGG11 for training from scratch ...\n")
        model_ft = embeddingVGG11(in_ch=3,num_classes=2)


elif train_mode=='transfer':
    # Load a pretrained model - MobilenetV2
    print("\nLoading mobilenetv2 as feature extractor ...\n")
    model_ft = models.mobilenet_v2(pretrained=True)    

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    # Modify fc layers to match num_classes
    num_ftrs=model_ft.classifier[-1].in_features
    model_ft.classifier=nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
        )    

# Transfer the model to device
model_ft = model_ft.to(device)

# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )
summary(model_ft, input_size=(3, 224, 128))
print(model_ft)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer 
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter("")
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_topk_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()
                #100 imagenes y el batch de 10
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # 10 predicciones correctos/100
                    #print(outputs)
                    _, preds = torch.max(outputs, 1)
                    
                    if len(class_names)>2:
                        _, top_k_preds = torch.topk(outputs,2)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if len(class_names)>2:
                    running_topk_corrects += torch.sum(top_k_preds[:,0] == labels.data) + torch.sum(top_k_preds[:,1] == labels.data)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if len(class_names)>2:
                epoch_acc_k = running_topk_corrects.double() / dataset_sizes[phase]
            else:
                epoch_acc_k = 0.0

            print('{} Loss: {:.4f} Acc: {:.4f} Top2Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,epoch_acc_k))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)

'''
Sample run: python train.py --mode=finetue
'''
