# Eval-probing
## Linguistic probing task?
Given an encoder model (e.g., BERT) pre-trained on a certain task, we use the representations it produces to trian a classifier (without further fine-tuning the model) to predict a linguistic property of the input text.
![image](https://s2.loli.net/2022/06/29/BZgLt79xhyjvIXu.png)
![Snipaste_2022-06-29_22-00-44.png](https://s2.loli.net/2022/06/29/fPydweCgAlmKbzF.png)

### Motivation of probe tasks?
- if we can train a classifier to predict a property of the input text based on its representation, it means the property is encoded somewhere in the representation;
- if we cannot train a classifier to predict a property of the input text based on its representation, it means the property is not encoded in the representation or not encoded in a useful way, considering how the representation is likely to be used;

### Probing approach
Literally, evaluation-based probing, it basically pulls out the output from the “module to probe”, attaches a prediction head (can be classification, regression, sequence labeling, and generation heads) on top of it.
At the time of probing, we randomly initialize the prediction head and train the prediction head with the weights in the “module to probe” frozen.
In real scenes, we compare the probing results between models or modules to assess the amount of task-relevant information stored in the models or modules.

![image](https://s2.loli.net/2022/06/29/RUtjIHlGbidDh4O.png)

## Run
### Requirements
```Python
transformers == 3.5.1
torch
tqdm
datasets
```
### Scripts
```Python
cd ./bertology/evaluation
python run_bert_classification.py --model_name_or_path ../../module/bert-base-uncased /
                                  --output_dir ../../output/bert_classification /
                                  --dataset SST2 /
                                  --epochs 5
```
