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
During training, we used the large model hyena for DNA embedding. To get the embedding, run the following file:

Hyena paper are accessed at:[Hyena Hierarchy:
 Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866)

### 2.2 Gene TPMs data and Gene read counts data
The data we downloaded from [GTEx Portal](https://www.gtexportal.org/home/)

More specifically, we downloaded EBV-transformed lymphocytes from [GTEx Portal Downloads](https://www.gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression)

## 3.Final result
