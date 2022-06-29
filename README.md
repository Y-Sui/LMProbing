# Eval-probing
## Linguistic probing task?
Given an encoder model (e.g., BERT) pre-trained on a certain task, we use the representations it produces to trian a classifier (without further fine-tuning the model) to predict a linguistic property of the input text.
![image](https://miro.medium.com/max/1260/1*7JDSKluZfSSI0O1yRWUIOQ.png)

### Motivation of probe tasks?
- if we can train a classifier to predict a property of the input text based on its representation, it means the property is encoded somewhere in the representation;
- if we cannot train a classifier to predict a property of the input text based on its representation, it means the property is not encoded in the representation or not encoded in a useful way, considering how the representation is likely to be used;

### Probing approach


## Requirements
```Python
transformers == 3.5.1
torch
tqdm
datasets
```
