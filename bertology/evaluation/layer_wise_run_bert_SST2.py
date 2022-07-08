import argparse
import os

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset.sst2.sst2 import SST2


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
        self.backbone = AutoModel.from_pretrained("bert-base-uncased")
        for p in self.parameters():
            p.requires_grad = False  # freeze the backbone model
        self.linear1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 2)  # 2 is the number of classes in this example
        self.__hidden_states__()

    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # backbone has the following shape: (batch_size, sequence_length, 768)
        # layer-wise
        layers = list(backbone.hidden_states)
        for i in range(len(backbone.hidden_states)):
            l1 = self.linear1(backbone.hidden_states[i])
            dropout = self.dropout(l1)
            l2 = self.linear2(dropout)
            layers[i] = l2
        return layers

    def __hidden_states__(self):
        backbone = self.backbone(torch.tensor([[1,1]]), torch.tensor([[1,1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states


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
        default="bert-base-uncased",
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
    elif args.dataset == "sst":  # will add later
        pass
    else:
        print("The dataset is empty!")
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=args.no_shuffle)
    eval_dataloader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=args.no_shuffle)
    # Reload the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)  # "../../module/bert-base-uncased"
    model = BertForClassification().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               model.parameters()))  # only update the parameters who are set to requires_grad
    # PATH = "../../output/bert_classification"
    output_model_file = args.output_dir + ".bin"

    if os.path.exists(output_model_file):
        # make sure the model_file is not empty
        model.load_state_dict(torch.load(output_model_file))
    else:
        print("\nThe model file is empty, train the model!\n")
        # add tqdm to the pipe
        # pbar_epochs = tqdm([f"Epochs-{i}" for i in range(1, args.epochs + 1)])
        for layer in tqdm(range(len(model.hidden_states))):
            # train the model
            for epoch in tqdm(range(1, args.epochs +1)):
                # pbar_epochs.set_description("Processing %s "%epoch)
                # model.train()
                for batch in tqdm(train_dataloader):
                    data = list(batch[0])
                    targets = torch.tensor(list(batch[1])).to(args.device)
                    optimizer.zero_grad()
                    # Style-1:
                    encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,
                                                           max_length=args.max_length,
                                                           add_special_tokens=True)
                    input_ids = encoding['input_ids'].to(args.device)
                    attention_mask = encoding['attention_mask'].to(args.device)

                    outputs = model(input_ids, attention_mask)
                    logits = outputs[layer][:, -1] # CLS head
                    predictions = torch.softmax(logits, dim=-1)

                    # Style-2:
                    # model_input = tokenizer(data, padding="max_length", max_length=args.max_length, truncation=True,
                    #                         return_tensors="pt")
                    # model_input.to(args.device)
                    # outputs = model(model_input["input_ids"], model_input["attention_mask"])
                    # logits = outputs
                    # predictions = torch.softmax(logits, dim=-1)

                    loss = criterion(predictions,
                                     targets)  # make sure the predictions and the targets are on the same device!
                    loss.backward()
                    optimizer.step()
                    
            # evaluate the model (no need to use the without_grad():)
            model.eval()
            glue_metric = datasets.load_metric('glue', 'sst2')  # load the metrics
            for batch in tqdm(eval_dataloader):
                data = list(batch[0])
                targets = torch.tensor(list(batch[1])).to(args.device)
                model_input = tokenizer(data, padding="max_length", max_length=args.max_length, truncation=True,
                                        return_tensors="pt")
                model_input.to(args.device)
                outputs = model(model_input["input_ids"], model_input["attention_mask"])
                logits = outputs[layer]
                predictions = torch.squeeze(torch.softmax(logits, dim=-1)).sum(0)
                model_predictions = []
                for i in range(len(targets)):
                    model_predictions.append(0) if predictions[i][0] > predictions[i][1] else model_predictions.append(
                        1)
                glue_metric.add_batch(predictions=model_predictions, references=targets)
            final_score = glue_metric.compute()
            with open(args.output_dir + "_layer_wise_final_score.txt", "a") as file:
                file.write(f"Accuracy of the Layer_{layer} is {final_score}"+"\n")

            torch.save(model.state_dict(), args.output_dir + f"layer_{layer}.bin")  # save the model


if __name__ == "__main__":
    main()
