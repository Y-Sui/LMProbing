from pathlib import Path

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset.sst2.sst2 import SST2
from dataset.sst.sst import SST

def save_model(model, dirpath):
    # save results
    if os.path.exists(dirpath):
        if os.path.exists(os.path.join(dirpath, "config.json")) and os.path.isfile(
            os.path.join(dirpath, "config.json")
        ):
            os.remove(os.path.join(dirpath, "config.json"))
        if os.path.exists(os.path.join(dirpath, "pytorch_model.bin")) and os.path.isfile(
            os.path.join(dirpath, "pytorch_model.bin")
        ):
            os.remove(os.path.join(dirpath, "pytorch_model.bin"))
    else:
        os.makedirs(dirpath)
    model.save_pretrained(dirpath)

class BertForClassification(nn.Module):
    def __init__(self):
        super(BertForClassification, self).__init__()
        self.backbone = AutoModel.from_pretrained("../../module/bert-base-uncased")
        for p in self.parameters():
            p.requires_grad = False  # freeze the backbone model
        self.linear1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 2)  # 2 is the number of classes in this example

    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids, attention_mask=attention_mask)
        # backbone has the following shape: (batch_size, sequence_length, 768)
        l1 = self.linear1(backbone[0])  # extract the last_hidden_state of the backbone model
        dropout = self.dropout(l1)
        l2 = self.linear2(dropout)
        return l2

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dataset",
        default="SST2",
        required=True,
        type=str,
        help="The dataset name, the options can be sst, SST2, etc"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="../../module/bert-base-uncased",
        type=str,
        required=True,
        help="Path to save the pretrained model"
    )
    parser.add_argument(
        "--output_dir",
        default="../../output/bert_classification",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    # Options parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--no_shuffle", action="store_true", help="Whether not to shuffle the dataloader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
    args = parser.parse_args()

    # Setup devices (No distributed training here)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # Print/save training arguments
    # os.makedirs(args.output_dir, exist_ok=True)
    # torch.save(args, os.path.join(args.output_dir, "run_args.bin"))

    # Prepare dataset
    training_data, evaluation_data = [], []
    if args.dataset == "SST2":
        training_data = SST2("train")
        evaluation_data = SST2("valid")
    elif args.dataset == "sst": # will add later
        pass
    else:
        print("The dataset is empty!")
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=args.no_shuffle)
    eval_dataloader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=args.no_shuffle)
    # Reload the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path) # "../../module/bert-base-uncased"
    model = BertForClassification().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()))  # only update the parameters who are set to requires_grad
    # PATH = "../../output/bert_classification"
    output_model_file = args.output_dir + "bin"

    if os.path.exists(output_model_file):
        # make sure the model_file is not empty
        model.load_state_dict(torch.load(output_model_file))
    else:
        print("The model file is empty, train the model!")
        # add tqdm to the pipe
        for epoch in tqdm(range(args.epochs)):
            # train the model
            model.train()
            for batch in tqdm(train_dataloader):
                data = list(batch[0])
                targets = torch.tensor(list(batch[1])).to(args.device)

                optimizer.zero_grad()
                encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,
                                                       max_length=args.max_length,
                                                       add_special_tokens=True)

                input_ids = encoding['input_ids'].to(args.device)
                attention_mask = encoding['attention_mask'].to(args.device)

                outputs = model(input_ids, attention_mask)
                logits = outputs[:, -1]
                predictions = torch.softmax(logits, dim=-1)

                loss = criterion(predictions, targets) # make sure the predictions and the targets are on the same device!
                loss.backward()
                optimizer.step()
            # evaluate the model
            model.eval()
            glue_metric = datasets.load_metric('glue', 'sst2') # load the metrics
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    data = list(batch[0])
                    targets = torch.tensor(list(batch[1])).to(args.device)

                    optimizer.zero_grad()
                    encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,
                                                           max_length=args.max_length,
                                                           add_special_tokens=True)

                    input_ids = encoding['input_ids'].to(args.device)
                    attention_mask = encoding['attention_mask'].to(args.device)

                    outputs = model(input_ids, attention_mask)
                logits = outputs[:, -1]
                model_predictions = torch.softmax(logits, dim=-1)
                glue_metric.add_batch(predictions=model_predictions, references=targets)
            final_score = glue_metric.compute()
            print(final_score)

        # torch.save(model.state_dict(), args.output_dir + ".bin")  # save the model


    # test the model
    input = "Pretty much sucks, but has a funny moment or two"
    model_input = tokenizer(input, padding="max_length", max_length=args.max_length, truncation=True, return_tensors="pt")
    model_input.to(args.device)
    outputs = model(model_input["input_ids"], model_input["attention_mask"])
    logits = outputs
    predictions = torch.squeeze(torch.softmax(logits, dim=-1)).sum(0)
    labels = "negative" if predictions[0] > predictions[1] else "positive"
    print(labels)
if __name__ == "__main__":
    main()
