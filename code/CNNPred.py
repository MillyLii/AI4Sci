import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load tensor
file_path = "../data/train.h5"
# 打开 HDF5 文件
with h5py.File(file_path, 'r') as file:
    data_halflife = file['halflife'][...]  
    data_label = file['label'][...]
        
    tensor_halflife_train = torch.tensor(data_halflife)
    tensor_label_train = torch.tensor(data_label)
    
file_path = "../data/valid.h5"

with h5py.File(file_path, 'r') as file:
    data_halflife = file['halflife'][...] 
    data_label = file['label'][...]
        
    tensor_halflife_valid = torch.tensor(data_halflife)
    tensor_label_valid = torch.tensor(data_label)

# Load embedding
load_path = '../tensor/tensor_promoter_train_embedding.pth'
tensor_promoter_train = torch.load(load_path,weights_only=False)

load_path = '../tensor/tensor_promoter_valid_embedding.pth'
tensor_promoter_valid = torch.load(load_path,weights_only=False)

# Define model
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        
        self.input_size = None
        
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 8, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv1d(8, 4, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2, 2)
        
        self.dropout = nn.Dropout(0.5) 
        
        if self.input_size is not None:
            self.fc1 = nn.Linear(self.input_size, 256)
        else:
            self.fc1 = None
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1) 
        
        self.fc_y = nn.Linear(256,128)
        self.linear_y = nn.Linear(128,1)
    
    def forward(self, x, y):
        y = nn.functional.relu(self.fc_y(y)) 
        y = self.linear_y(y) 
        
        x = x.unsqueeze(2)

        concat_tensor = torch.concat((x,y),dim=1)
        concat_tensor = concat_tensor.unsqueeze(1)
        concat_tensor = concat_tensor.squeeze(-1)
        
        concat_tensor = self.pool(nn.functional.relu(self.conv1(concat_tensor)))
        concat_tensor = self.pool(nn.functional.relu(self.conv2(concat_tensor)))
        concat_tensor = self.pool(nn.functional.relu(self.conv3(concat_tensor)))  
        concat_tensor = self.pool(nn.functional.relu(self.conv4(concat_tensor)))  
        
        concat_tensor = concat_tensor.view(concat_tensor.size(0), -1)
        
        if self.fc1 is None:  
            self.input_size = concat_tensor.size(1)  
            self.fc1 = nn.Linear(self.input_size, 256).to(concat_tensor.device)  
        
        concat_tensor = nn.functional.relu(self.fc1(concat_tensor))
        concat_tensor = self.dropout(concat_tensor) 
        concat_tensor = nn.functional.relu(self.fc2(concat_tensor))
        logit = self.fc3(concat_tensor) 
        
        return logit
    
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

model = CNNBinaryClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

losses = []

# 训练模型
num_epochs = 50  
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_data1, batch_data2, batch_labels in dataloader:
        batch_labels = batch_labels.float().unsqueeze(1).to(device)
        
        batch_data1 = batch_data1.float().to(device)# halflife
        batch_data2 = batch_data2.float().to(device)# promoter

        optimizer.zero_grad()

        outputs = model(batch_data1,batch_data2)
        
        loss = criterion(outputs, batch_labels)
        
        loss.backward()
        optimizer.step()

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

# 保存图像
plt.savefig(f'../50epochsCNN.png', bbox_inches='tight')
plt.close()

save_path = '../model/CNNpred.pth' # 记得改名字
torch.save(model.state_dict(), save_path)

# Validate model
model.to('cpu')
tensor_halflife_valid = tensor_halflife_valid.float().to('cpu')
tensor_promoter_valid = tensor_promoter_valid.float().to('cpu')

model.eval()

with torch.no_grad():
    outputs = model(tensor_halflife_valid,tensor_promoter_valid)

probabilities = torch.sigmoid(outputs)

valid_labels = tensor_label_valid.float()

predicted_labels = (probabilities > 0.55).float() 

# 计算各项指标
accuracy = accuracy_score(valid_labels.numpy(), predicted_labels.numpy())
precision = precision_score(valid_labels.numpy(), predicted_labels.numpy())
recall = recall_score(valid_labels.numpy(), predicted_labels.numpy())
f1 = f1_score(valid_labels.numpy(), predicted_labels.numpy())

auc = roc_auc_score(valid_labels.numpy(), probabilities.numpy())

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC: {auc:.4f}')
