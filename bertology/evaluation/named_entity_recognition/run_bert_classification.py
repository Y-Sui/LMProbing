import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import BertForTokenClassification, BertTokenizer

import datasets
from datasets import load_dataset, load_metric

datasets = load_dataset("conll2003")

task = "ner" # should be one of "ner", "pos" or "chunk"

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
        self.backbone = AutoModel.from_pretrained("distilbert-base-uncased")
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
    parser.add_argument("--dataset",default="SST2",type=str,help="The dataset name, the options can be sst, SST2, etc")
    parser.add_argument("--model_name_or_path",default="bert-base-uncased",type=str,help="Path to save the pretrained model")
    parser.add_argument("--output_dir",default="../../output/bert_classification",type=str,help="The output directory where the model predictions and checkpoints will be written.")
    # Options parameters
    parser.add_argument("--config_name",default="",type=str,help="Pretrained config name or path if not the same as model_name_or_path",)
    parser.add_argument("--tokenizer_name",default="",type=str,help="Pretrained tokenizer name or path if not the same as model_name_or_path",)
    parser.add_argument("--cache_dir",default=None,type=str,help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--no_shuffle", action="store_true", help="Whether not to shuffle the dataloader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
    args = parser.parse_args()

    # Setup devices (No distributed training here)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Prepare dataset
    training_data, evaluation_data = [], []
    training_data = SST2("train")
    evaluation_data = SST2("valid")
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=args.no_shuffle)
    eval_dataloader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=args.no_shuffle)
    # Reload the model
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = BertForTokenClassification.from_pretrained(args.model_name_or_path).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    print("\nThe model file is empty, train the model!\n")
    # add tqdm to the pipe
    for _ in tqdm(range(1, args.epochs + 1)):
        # train the model
        model.train()
        for batch in tqdm(train_dataloader):
            data = list(batch[0])
            targets = torch.tensor(list(batch[1])).float().to(args.device)
            # print(targets)
            optimizer.zero_grad()
            model_input = tokenizer(data, padding="max_length", max_length=args.max_length, truncation=True,
                                    return_tensors="pt")
            input_ids = model_input['input_ids'].to(args.device)
            attention_mask = model_input['attention_mask'].to(args.device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits[:,0,:] # CLS
            # print(logits)
            predictions = torch.argmax(logits, dim=-1).float()
            # print(predictions)
            loss = criterion(predictions, targets)  # make sure the predictions and the targets are on the same device!
            loss.requires_grad = True
            loss.backward()
            print(loss)
            optimizer.step()
    # evaluate the model (no need to use the without_grad():)
    model.eval()
    glue_metric = datasets.load_metric('glue', 'sst2')  # load the metrics
    for batch in tqdm(eval_dataloader):
        data = list(batch[0])
        targets = torch.tensor(list(batch[1])).float().to(args.device)
        model_input = tokenizer(data, padding="max_length", max_length=args.max_length, truncation=True,
                                return_tensors="pt")
        model_input.to(args.device)
        outputs = model(model_input["input_ids"], model_input["attention_mask"])
        logits = outputs.logits[:,0,:] # CLS
        predictions = torch.argmax(logits, dim=-1).float()
        glue_metric.add_batch(predictions=predictions, references=targets)
    final_score = glue_metric.compute()
    print(final_score)


if __name__ == "__main__":
    main()
