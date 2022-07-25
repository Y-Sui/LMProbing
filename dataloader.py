from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import torch

TASK = "ner"
ptm="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ptm)

def load_dataset_huggingface(dataset_name="wnut_17"):
    dataset = load_dataset(dataset_name)
    return dataset

def tokenization(example):
    return tokenizer(example["tokens"])

def tokenize_and_align_labels(example):

    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=50) # is_split_into_words=True, Whether or not the input is already pre-tokenized (e.g., split into words)
    labels = []
    for i, label in enumerate(example[f"{TASK}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return  tokenized_inputs

def construct_data_loader(batch_size, shuffle=True, num_workers=0):
    wnut = load_dataset_huggingface("wnut_17")
    wnut = wnut.map(tokenize_and_align_labels, batched=True)
    # Set the format of your dataset to be compatible with your machine learning framework:
    wnut.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    wnut_train_set, wnut_eval_set, wnut_test_set = wnut["train"], wnut["validation"], wnut["test"]
    wnut_train_dataloader = DataLoader(wnut_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    wnut_eval_dataloader = DataLoader(wnut_eval_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    wnut_test_dataloader = DataLoader(wnut_test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    label_list = wnut["train"].features[f"{TASK}_tags"].feature.names

    return wnut_train_dataloader, \
           wnut_eval_dataloader, \
           wnut_test_dataloader, \
           label_list

# # wnut = load_dataset_huggingface("wnut_17")
# # wnut = wnut.map(tokenize_and_align_labels, batched=True)
# # wnut.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# # wnut_train = wnut['train']
# # print(wnut_train[0])
# # print(wnut_train[1])

# # dataloader = torch.utils.data.DataLoader(wnut_train, batch_size=32)
# # print(next(iter(dataloader)))


# wnut_train_dataloader, \
# wnut_eval_dataloader, \
# wnut_test_dataloader, \
# wnut_label_list = construct_data_loader(batch_size=16, shuffle=True)

# for example_batched in wnut_eval_dataloader:
#     input_ids = example_batched["input_ids"]
#     attention_mask = example_batched["attention_mask"]
#     labels = example_batched["labels"]
#     print(input_ids, attention_mask, labels)