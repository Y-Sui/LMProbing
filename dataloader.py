from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from run import args

TASK = args.task
ptm="bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ptm)

def load_dataset_huggingface(dataset_name="wnut_17"):
    dataset = load_dataset(dataset_name)
    return dataset

def load_dataset_json(filePath="ner"):
    dataset = load_dataset("json", data_files={"train": f"dataset/{filePath}_train.json",
                                               "valuation": f"dataset/{filePath}_eval.json"}, field="data")
    return dataset

def load_dataset_single_json(category="ner", filePath="wsj_annotated_ner_LOC"):
    dataset = load_dataset("json", data_files={"train": f"dataset/{category}/{filePath}_train.json",
                                               "validation": f"dataset/{category}/{filePath}_eval.json"}, field= "data")
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

def construct_data_loader_huggingface(batch_size, shuffle=True, num_workers=0):
    """
    construct dataloader from huggingface
    """
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


def construct_data_loader(batch_size, dataset="ner", filePath="wsj_annotated_ner_LOC", shuffle=True, num_workers=0):
    """
    construct dataloader from custom dataset
    """
    # probing_dataset = load_dataset_json(dataset) # load mixed-dataset
    probing_dataset = load_dataset_single_json(dataset, filePath) # load single dataset
    probing_dataset = probing_dataset.map(tokenize_and_align_labels, batched=True)

    # Set the format of your dataset to be compatible with your machine learning framework:
    probing_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    probing_train_set, probing_eval_set = probing_dataset["train"], probing_dataset["validation"]
    probing_train_dataloader = DataLoader(probing_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    probing_eval_dataloader = DataLoader(probing_eval_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Load the label_list (mixed version)
    # if dataset == "ner":
    #     tag_dict = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-MISC": 5, "I-MISC": 6, "B-LOC": 7,
    #                 "I-LOC": 8}
    # elif dataset == "chunk":
    #     tag_dict = {"O": 0, "B-ADJP": 1, "I-ADJP": 2, "B-ADVP": 3, "I-ADVP": 4, "B-CONJP": 5, "I-CONJP": 6, "B-INTJ": 7,
    #                 "I-INTJ": 8, "B-LST": 9, "I-LST": 10, "B-NP": 11, "I-NP": 12, "B-PP": 13, "I-PP": 14, "B-PRT": 15, "I-PRT": 16,
    #                 "B-SBAR": 17, "I-SBAR": 18, "B-VP": 19, "I-VP": 20}
    # else:
    #     tag_dict = {}

    # Load the label_list (single samples)
    category = filePath.split("_")[-1]
    tag_dict = {"O": 0, f"B-{category}": 1, f"I-{category}": 2}
    if category.islower():
        tag_dict = {"O": 0, "B-head": 1, "B-dependent": 2}

    return probing_train_dataloader, probing_eval_dataloader, list(tag_dict.keys())

# if __name__ == "__main__":
#     a, b, c = construct_data_loader(2)
#     print(c)
