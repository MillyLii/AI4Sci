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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load tensor
file_path = "../data/train.h5"
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
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),  # 输入通道4 -> 输出通道16
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),  # 输入通道4 -> 输出通道16
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),  # 输出通道16 -> 8，降维
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),  # 输出通道8 -> 1，降维
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # 解码器（可以省略，主要用作监督学习）
        self.decoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 输出和原始维度一致
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
# Train model
class promoterDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        return sample
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input = tensor_promoter_train.float()
input = input.permute(0,2,1)

train_dataset = promoterDataset(input)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = Autoencoder().to(device)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)

losses = []

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:  
        
        torch.cuda.empty_cache()
        
        batch = batch.to(device)  
        optimizer.zero_grad()

        encoded_train, decoded_train = model(batch)  
        loss = criterion(decoded_train, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

    losses.append(avg_loss)
    
# 绘制损失图像
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.to('cpu')
model.eval()

input_valid = tensor_promoter_valid.float()
input_valid = input_valid.permute(0,2,1)

with torch.no_grad():
    encoded_valid,decoded_valid = model(input_valid)
    
error = criterion(decoded_valid,input_valid)
print(error)

save_path = 'model/AEEncoder.pth'
torch.save(model.state_dict(), save_path)

# Using XGBoost
input = tensor_promoter_train.float()
input = input.permute(0,2,1)
input = input.to('cpu')
model.to('cpu')
encoded_train,_ = model(input)

encoded_train = encoded_train.squeeze(1)

XGBinput_train = torch.cat((tensor_halflife_train,encoded_train),dim=1)

model = xgb.XGBClassifier(objective='binary:logistic',  
                            eval_metric='logloss',         
                            n_estimators=100,              
                            max_depth=7,                   
                            learning_rate=0.1,             
                            subsample=0.8,                 
                            colsample_bytree=0.8           
)
model.fit(XGBinput_train.detach().numpy(),tensor_label_train.detach().numpy())

encoded_valid = encoded_valid.squeeze(1)
XGBinput_valid = torch.cat((tensor_halflife_valid,encoded_valid),dim=1)

predictions = model.predict(XGBinput_valid.detach().numpy())

# 计算各项指标
accuracy = accuracy_score(tensor_label_valid.numpy(), predictions)
precision = precision_score(tensor_label_valid.numpy(), predictions)
recall = recall_score(tensor_label_valid.numpy(), predictions)
f1 = f1_score(tensor_label_valid.numpy(), predictions)

auc = roc_auc_score(tensor_label_valid.numpy(), predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC: {auc:.4f}')

model.save_model('model/xgboost_model.bin')