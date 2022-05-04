import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import confusion_matrix, classification_report
from net import BinaryClassification
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

## test data    
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

#test_data = TestData(torch.FloatTensor(X_test))

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

#train_data = TrainData(torch.FloatTensor(X_train), 
#                       torch.FloatTensor(y_train))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = BinaryClassification()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(dataset=test_data, batch_size=1)

train_data = np.load('employee_embeddings.npy')
train_labels = np.load('employee_labels.npy')

tensor_x = torch.Tensor(train_data) # transform to torch tensor
tensor_y = torch.Tensor(train_labels)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE) # create your dataloader

model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')



#TEST

test_data = np.load('val_employee_embeddings.npy')
test_labels = np.load('val_employee_labels.npy')

test_data = TestData(torch.FloatTensor(test_data))
#test_x = torch.Tensor(train_data) # transform to torch tensor


#my_dataset = TensorDataset(test_x) # create your datset
test_loader = DataLoader(test_data, batch_size=1) # create your dataloader

y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


confusion_matrix(test_labels, y_pred_list)

print(classification_report(test_labels, y_pred_list))
