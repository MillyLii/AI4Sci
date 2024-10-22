# AI4Sci 基于DNA序列信息预测基因表达

## 1.Install
### 1.1 Requirements
python >= 3.10.12; Pytorch == 2.4.1; CUDA = 12.1

### 1.2 Quickstart
```
conda create -n AIDNA python==3.10.12 -y
conda activate AIDNA
pip install -r requirements.txt
```
## 2.Data Source
### 2.1 DNA Embedding
During training, we used the large model hyena for DNA embedding.

To get the embedding, run the following file:

*Please note that it is important to get the embedding before training, validating, and testing!*

Once the embedding is obtained, the following code can be executed to obtain the tensor for use in the code:
```python
python

>>> import numpy as np
>>> import torch

>>> embedding_promoter_train = np.load('data/embedding/sequence_hyena_train.npy')
>>> tensor_promoter_train = torch.tensor(embedding_promoter_train)

>>> embedding_promoter_valid = np.load('data/embedding/sequence_hyena_valid.npy')
>>> tensor_promoter_valid = torch.tensor(embedding_promoter_valid)

>>> embedding_promoter_test = np.load('data/embedding/sequence_hyena_test.npy')
>>> tensor_promoter_test = torch.tensor(embedding_promoter_test)
```

Hyena paper are accessed at:[Hyena Hierarchy:
 Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866)

### 2.2 Gene TPMs data and Gene read counts data
The data we downloaded from [GTEx Portal](https://www.gtexportal.org/home/)

More specifically, we are downloading data related to EBV-transformed lymphocytes  from [GTEx Portal Downloads](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression)

## 3.Training
### 3.1 Train a predictive model using CNN
To train a CNN-based predictive model, please run:
```
python CNNPred.py
```
### 3.2 Train a predictive model using LSTM
To train a LSTM-based predictive model, please run:
```
python LSTMPred.py
```
### 3.3 Train a predictive model using AutoEncoder and XGBoost
To train AE-XGB-based predictive model, please run:
```
python AEXGBPred.py
```
### 3.4 Train a predictive model using extra features
using Gene read counts as extra feature, please run:
```
python CNNWithReadCountsPred.py
```
using Gene TPMs as extra feature, please run:
```
python CNNWithTPMPred.py
```
## 4. Final results
