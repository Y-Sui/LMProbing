import os

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
import torch
import logging

from dataset.config import DataConfig
from torch.utils.data.distributed import DistributedSampler

import torch.distributed as dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_argument():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ner", type=str, help="Please specify the task name {NER or Chunk}")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="Path to save the pretrained model")
    parser.add_argument("--embed_size", default=256, type=int)
    parser.add_argument("--label_size", default=2, type=int,
                        help="classification task: the number of the label classes")
    parser.add_argument("--corpus", default="//home/weicheng/data_interns/yuan/", type=str)
    # Options parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name_or_path", )
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name_or_path", )
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--no_shuffle", action="store_true", help="Whether not to shuffle the dataloader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--profile", action="store_true", help="whether to generate the heatmap")
    parser.add_argument("--mode", choices=["layer-wise", "head-wise"], type=str, help="choose training mode")
    args = parser.parse_args()

    # Setup devices (No distributed training here)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    return args


def get_files_path(filePath, outPath):
    """
    return the file list
    """
    raw_files = os.listdir(filePath)
    out_files = os.listdir(outPath)
    file_paths = {}
    train_files, eval_files = [], []
    for i in range(len(raw_files)):
        if raw_files[i].__contains__("train") == 1:
            train_files.append(os.path.join(filePath, raw_files[i]))
            for j in range(len(raw_files)):
                if raw_files[j].__contains__("eval") and \
                        raw_files[j].replace("eval", "") == raw_files[i].replace("train", ""):
                    eval_files.append(os.path.join(filePath, raw_files[j]))

    file_paths["train"], file_paths["eval"] = train_files, eval_files
    return file_paths


def get_tag_dict(category):
    tag_dict = {"O": 0, f"B-{category}": 1, f"I-{category}": 2}
    if category.islower():
        tag_dict = {"O": 0, "B-head": 1, "B-dependent": 2}
    if category == "VP":
        tag_dict = {"O": 0, "B-ADVP": 1, "I-ADVP": 2, "B-VP": 3, "I-VP": 4}
    if category == "":
        tag_dict = {"O": 0, "B-:": 1, "I-:": 2}
    if category == "EX":
        tag_dict = {"O": 0, "B-E:": 1, "I-E:": 2}
    if category == "TO":
        tag_dict = {"O": 0, "B-T:": 1, "I-T:": 2}
    return list(tag_dict.keys())


class Sequence_Classification:

    def __init__(self, args):
        self.task = args.task
        self.ptm = args.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.ptm)
        self.max_length = args.max_length
        self.logger = logging.getLogger("Sequence classification")
        self.logger.info("start dataloader generation")

    def tokenize_and_align_labels(self, example):
        tokenized_inputs = self.tokenizer(
            example["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_length
        )  # is_split_into_words=True, whether or not the input is already pre-tokenized
        labels = []
        for i, label in enumerate(example[f"{self.task}_tags"]):
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
        return tokenized_inputs

    def construct_data_loader(self, batch_size, idx, file_path, shuffle=True, num_workers=4, rank=0):
        corpus = load_dataset("json",
                              field="data",
                              data_files={
                                  "train": file_path["train"][idx],
                                  "validation": file_path["eval"][idx]})

        self.logger.info("map each tokenization to its word_ids and align with labels")
        probing_dataset = corpus.map(self.tokenize_and_align_labels, batched=True)
        # Set the format of your dataset to be compatible with your machine learning framework:
        probing_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        probing_train_dataset, probing_valid_dataset = probing_dataset["train"], probing_dataset["validation"]
        probing_train_dataloader = DataLoader(probing_train_dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers)
        probing_eval_dataloader = DataLoader(probing_valid_dataset, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers)

        # Load the label_list (single samples)
        category = file_path["train"][idx].split("_")[-2]
        self.logger.info(f"get the {category} corresponding tag dict")
        tag_dict = get_tag_dict(category)

        return probing_train_dataloader, probing_eval_dataloader, tag_dict


def get_sequence_classification(idx, flag):
    args = get_argument()
    sample_config = DataConfig()
    pos_sample_path = get_files_path(filePath=os.path.join(sample_config.data_path, f"{args.task}", "samples"),
                                     outPath=os.path.join(sample_config.output_path, f"{args.task}"))
    neg_sample_path = get_files_path(filePath=os.path.join(sample_config.data_path, f"{args.task}", "neg_samples"),
                                     outPath=os.path.join(sample_config.output_path, f"{args.task}"))
    if flag == "pos":
        dataloader_config = Sequence_Classification(args)
        probing_train_dataloader, \
        probing_eval_dataloader, \
        probing_label_list = dataloader_config.construct_data_loader(batch_size=args.batch_size, idx=idx,
                                                                     file_path=pos_sample_path,
                                                                     shuffle=True if not args.no_shuffle else True,
                                                                     num_workers=args.num_workers)
        return probing_train_dataloader, probing_eval_dataloader, probing_label_list
    else:
        dataloader_config = Sequence_Classification(args)
        neg_probing_train_dataloader, \
        neg_probing_eval_dataloader, \
        neg_probing_label_list = dataloader_config.construct_data_loader(batch_size=args.batch_size, idx=idx,
                                                                         file_path=neg_sample_path,
                                                                         shuffle=True if not args.no_shuffle else True,
                                                                         num_workers=args.num_workers)
        return neg_probing_train_dataloader, neg_probing_eval_dataloader, neg_probing_label_list


def main():
    args = get_argument()
    sample_config = DataConfig()
    pos_sample_path = get_files_path(filePath=os.path.join(sample_config.data_path, f"{args.task}", "samples"),
                                     outPath=os.path.join(sample_config.output_path, f"{args.task}"))
    neg_sample_path = get_files_path(filePath=os.path.join(sample_config.data_path, f"{args.task}", "neg_samples"),
                                     outPath=os.path.join(sample_config.output_path, f"{args.task}"))
    for i in range(len(pos_sample_path["train"])):
        a, b, c = get_sequence_classification(i, "pos")


if __name__ == "__main__":
    main()
