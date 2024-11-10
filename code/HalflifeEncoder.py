import h5py
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = "../data/train.h5"
with h5py.File(file_path, 'r') as file:
    data_halflife = file['halflife'][...]
    data_label = file['label'][...]
        
    tensor_halflife_train = torch.tensor(data_halflife)
    tensor_label_train = torch.tensor(data_label)
    
    
tensor_halflife_train = tensor_halflife_train.float().to(device)
    
load_path = '../tensor/tensor_promoter_train_embedding.pth'
tensor_promoter_train = torch.load(load_path)

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, out_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size,padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=kernel_size,padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=out_channels, kernel_size=kernel_size,padding=1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x
    
def augment_data(data, noise_level=0.1):
    noise = torch.randn_like(data) * noise_level
    return data + noise

def get_random_negative_sample(data_loader):
    random_batch = random.choice(list(data_loader))

    negative_sample = random.choice(random_batch)
    
    return negative_sample  

def train(model, data_loader, optimizer, num_epochs):
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2) 
    model.train()  
    for epoch in range(num_epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()

            augmented_inputs2 = augment_data(data, 0.1)
            negative_sample = get_random_negative_sample(data_loader)
            negative_sample = negative_sample.unsqueeze(0)

            anchor_output = model(data)
            pos_output = model(augmented_inputs2)
            neg_output = model(negative_sample)

            loss = triplet_loss(anchor_output,pos_output,neg_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader):.4f}')

data_loader = torch.utils.data.DataLoader(tensor_halflife_train, batch_size=32, shuffle=True)

input_dim = 1
hidden_dim = 128
kernel_size = 3
out_channels = 256

model = CNNEncoder(input_dim, hidden_dim, kernel_size, out_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

train(model, data_loader, optimizer, num_epochs=10)

save_path = '../model/CNNHFEncode.pth' 
torch.save(model.state_dict(), save_path)