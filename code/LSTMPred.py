import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load tensor
file_path = "../data/train.h5"
# 打开 HDF5 文件
with h5py.File(file_path, 'r') as file:
    data_halflife = file['halflife'][...]  
    data_promoter = file['promoter'][...]
    data_label = file['label'][...]
        
    tensor_halflife_train = torch.tensor(data_halflife)
    tensor_promoter_train = torch.tensor(data_promoter)
    tensor_label_train = torch.tensor(data_label)
    
file_path = "../data/valid.h5"

with h5py.File(file_path, 'r') as file:
    data_halflife = file['halflife'][...] 
    data_promoter = file['promoter'][...]
    data_label = file['label'][...]
        
    tensor_halflife_valid = torch.tensor(data_halflife)
    tensor_promoter_valid = torch.tensor(data_promoter)
    tensor_label_valid = torch.tensor(data_label)

# Load embedding
load_path = '../tensor/tensor_promoter_train_embedding.pth'
tensor_promoter_train = torch.load(load_path,weights_only=False)

load_path = '../tensor/tensor_promoter_valid_embedding.pth'
tensor_promoter_train = torch.load(load_path,weights_only=False)

# Define model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.fc_y = nn.Linear(256,128)
        self.linear1_y = nn.Linear(128,64)
        self.linear2_y = nn.Linear(64,1)

    def forward(self, y):

        y = nn.functional.relu(self.fc_y(y))  
        y = self.linear1_y(y) 
        y = self.linear2_y(y)
        
        y = y.permute(0,2,1)
        
        lstm_out, (hn, cn) = self.lstm(y) 
        out = self.fc(hn[-1])  

        return out 
    
load_path = '../tensor/tensor_promoter_train_embedding.pth'
tensor_promoter_train = torch.load(load_path)

load_path = '../tensor/tensor_promoter_valid_embedding.pth'
tensor_promoter_valid = torch.load(load_path)

# Train model
class DNADataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        x = self.data1[idx]
        y = self.data2[idx]
        z = self.labels[idx]
        return x, y, z
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = DNADataset(tensor_halflife_train,tensor_promoter_train,tensor_label_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

input_size = 20002     
hidden_size = 1024 
num_classes = 1     

model = LSTMClassifier(input_size,hidden_size,num_classes).to(device)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)

losses = []

num_epochs = 10 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_data1, batch_data2, batch_labels in dataloader:

        batch_labels = batch_labels.float().unsqueeze(1).to(device)
        
        batch_data1 = batch_data1.float().to(device)# halflife
        batch_data2 = batch_data2.float().to(device)# promoter

        optimizer.zero_grad()

        outputs = model(batch_data2)
        
        # 计算损失
        loss = criterion(outputs, batch_labels)
        
        # print(loss)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 累加损失
        running_loss += loss.item()
        
    losses.append(running_loss/len(dataloader))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    
# 绘制损失图像
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save model
save_path = '../model/LSTMpred.pth'
torch.save(model.state_dict(), save_path)

# Validate model
model.to('cpu')

tensor_promoter_valid = tensor_promoter_valid.float().to('cpu')

model.eval()

with torch.no_grad():
    outputs = model(tensor_promoter_valid)

valid_labels = tensor_label_valid.float()

predicted_labels = (outputs > 0.55).float() 

accuracy = accuracy_score(valid_labels.numpy(), predicted_labels.numpy())
precision = precision_score(valid_labels.numpy(), predicted_labels.numpy())
recall = recall_score(valid_labels.numpy(), predicted_labels.numpy())
f1 = f1_score(valid_labels.numpy(), predicted_labels.numpy())

auc = roc_auc_score(valid_labels.numpy(), outputs.numpy())

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC: {auc:.4f}')