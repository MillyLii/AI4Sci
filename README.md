# AI4Sci 基于DNA序列信息预测基因表达

## 1.Install
### 1.1 Requirements
python >= 3.10.12; Pytorch == 2.4.1; CUDA = 12.1.

### 1.2 Quickstart
```
conda create -n AIDNA python==3.10.12 -y
conda activate AIDNA
pip install -r requirements.txt
```

## 2.Data Source
### 2.1 DNA Embedding
During training, we used the large model hyena for DNA embedding.

To get the embedding, run the following file: code/inference.py.

**NOTE**: The file path needs to be changed each time an embedding is performed.

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

For later use, run the following code to save the tensor:
```python
python

>>> torch.save(tensor_promoter_train,'tensor/tensor_promoter_train_embedding.pth')
>>> torch.save(tensor_promoter_valid,'tensor/tensor_promoter_valid_embedding.pth')
>>> torch.save(tensor_promoter_test,'tesnor/tensor_promoter_test_embedding.pth')
```

Hyena paper are accessed at: [Hyena Hierarchy:
 Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866).

## 3.Training
### 3.1 Train a predictive model using CNN
To train a CNN-based predictive model with half-life encoded data, please run:
```
python code/HalflifeEncoderCNN.py
```
To train a CNN-based predictive model, please run:
```
python code/CNNPred.py
```
### 3.2 Train a predictive model using LSTM
To train a LSTM-based predictive model, please run:
```
python code/LSTMPred.py
```
### 3.3 Train a predictive model using AutoEncoder and XGBoost
To train AE-XGB-based predictive model, please run:
```
python code/AEXGBPred.py
```
### 3.4 Train a predictive model using extra features
using Gene read counts as extra feature, please run:
```
python code/CNNWithReadCountsPred.py
```
using Gene TPMs as extra feature, please run:
```
python code/CNNWithTPMPred.py
```
### 3.5 using ready-made trained models
If you don't train, we also provide ready-made trained models, just load the model and use it!
To use the model, please run(use CNN model as example):
```python
python

# Define the model before using it (Detailed usage information is in the CNNtest.ipynb file)

>>> model =  CNNBinaryClassifier()

>>> model.load_state_dict(torch.load('../model/CNNPred.pth')

>>> model.eval()

```
## 4. Final results
### 4.1 Validation set
For CNN-based model with encoded halflife, we get:
 *Accuracy*: 0.8301, 
 *Precision*: 0.8385, 
 *Recall*: 0.8182, 
 *F1 Score*: 0.8282, 
 *AUC*: 0.8999.
 
For CNN-based model, we get:
*Accuracy*: 0.7998, 
 *Precision*: 0.7676, 
 *Recall*: 0.8606, 
 *F1 Score*: 0.8114, 
 *AUC*: 0.8662.

For AE-XGB-based model, we get:
*Accuracy*: 0.7755, 
 *Precision*: 0.7442, 
 *Recall*: 0.8404, 
 *F1 Score*: 0.7894, 
 *AUC*: 0.7755.

For LSTM-based model, we get:
*Accuracy*: 0.7290, 
 *Precision*: 0.7162, 
 *Recall*: 0.7596, 
 *F1 Score*: 0.7373， 
 *AUC*: 0.7861.

### 4.2 Test set
For CNN-based model with encoded halflife, we get:
 *Accuracy*: 0.8212, 
 *Precision*: 0.8275, 
 *Recall*: 0.8125, 
 *F1 Score*: 0.8199, 
 *AUC*: 0.8965.
 
For CNN-based model, we get:
*Accuracy*: 0.8182, 
 *Precision*: 0.8004, 
 *Recall*: 0.8488, 
 *F1 Score*: 0.8239, 
 *AUC*: 0.8822.

For AE-XGB-based model, we get:
*Accuracy*: 0.7667, 
 *Precision*: 0.7405, 
 *Recall*: 0.8226, 
 *F1 Score*: 0.7794, 
 *AUC*: 0.7666.
 
For LSTM-based model, we get:
*Accuracy*: 0.7545, 
 *Precision*: 0.7545, 
 *Recall*: 0.7560, 
 *F1 Score*: 0.7553， 
 *AUC*: 0.8233.
