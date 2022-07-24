import argparse
import os
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from dataset.sst2.sst2 import SST2


class BertForClassification(nn.Module):
    def __init__(self):
        super(BertForClassification, self).__init__()
        self.backbone = BertModel.from_pretrained("bert-base-uncased")
        self.num_heads = self.backbone.config.num_attention_heads
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze the backbone model
        self.linear1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 2)  # 2 is the number of classes in this example
        self.__hidden_states__()

    def forward(self, input_ids, attention_mask):
        head_mask = [0 for _ in range(self.num_heads)]
        head_mask_list = []
        for i in range(len(head_mask)):
            head_mask_n = head_mask[:]
            head_mask_n[i] = 1
            head_mask_list.append(head_mask_n)
        head_mask_list = torch.tensor(head_mask_list).to("cuda")
        head_hidden_states = torch.rand(self.num_heads, attention_mask.shape[0],attention_mask.shape[1], 2)
        for i in range(len(head_mask_list)):
            backbone_n = self.backbone(input_ids, attention_mask=attention_mask, head_mask=head_mask_list[i])
            l1 = self.linear1(backbone_n.last_hidden_state)
            dropout = self.dropout(l1)
            l2 = self.linear2(dropout)
            head_hidden_states[i] = l2
        return head_hidden_states


    def __hidden_states__(self):
        backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states

# save function
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


# train function
def train(epochs, trainLoader, model, tokenizer, max_length, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               model.parameters()))  # only update the parameters who are set to requires_grad
    output_path = "../../output/bert_classification_head_wise/"
    for head in tqdm(range(model.num_heads)):
        # train the model
        for epoch in tqdm(range(1, epochs + 1)):
            for i, batch in enumerate(tqdm(trainLoader)):
                data = list(batch[0])
                targets = torch.tensor(batch[1]).float().to(device)
                optimizer.zero_grad()
                encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True,
                                                       max_length=max_length,
                                                       add_special_tokens=True)
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                logits = outputs[head,:,0,:]  # CLS
                predictions = torch.argmax(logits, dim=-1).float().to(device)
                # make sure the predictions and the targets are on the same device!
                loss = criterion(predictions, targets)
                loss.requires_grad = True
                loss.backward()
                optimizer.step()
                if i % 200 == 0:
                    print(f"epoch: {epoch}, batch: {i}, loss: {loss.data}")
        torch.save(model, output_path + f"bert_classification_head_{head}.bin")


# test function
def test(head, eval_dataloader, model, tokenizer, max_length, device):
    model.to(device)
    with torch.no_grad(): # when in test stage, no grad
        glue_metric = datasets.load_metric('glue', 'sst2')  # load the metrics
        for batch in tqdm(eval_dataloader):
            data = list(batch[0])
            targets = torch.tensor(batch[1]).to(device)
            model_input = tokenizer(data, padding="max_length", max_length=max_length, truncation=True,
                                    return_tensors="pt")
            model_input.to(device)
            outputs = model(model_input["input_ids"], model_input["attention_mask"])
            logits = outputs[head,:,0,:]  # CLS
            predictions = torch.argmax(logits, dim=-1).to(device)
            # for i in range(len(targets)):
            #     model_predictions.append(0) if predictions[i][0] > predictions[i][1] else model_predictions.append(1)
            glue_metric.add_batch(predictions=predictions, references=targets)
        final_score = glue_metric.compute()
        with open("../../output/" + "bert_classification_head_wise_final_score.txt", "a") as file:
            file.write(f"Accuracy of the head_{head} is {final_score}" + "\n")


# predict function
def predict(input, model, device):
    pass

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", default="SST2", required=True, type=str, help="The dataset name, the options can be sst, SST2, etc")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str, required=True, help="Path to save the pretrained model")
    parser.add_argument("--output_dir", default="../../output/bert_classification", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    # Options parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name_or_path",)
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default=None, type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true", default=False, help="Whether not to shuffle the dataloader")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the tokenization")
    args = parser.parse_args()

    # Setup devices (No distributed training here)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Prepare dataset
    training_data, evaluation_data = [], []
    if args.dataset == "SST2":
        training_data, evaluation_data = SST2("train"), SST2("valid")
    elif args.dataset == "sst":  # will add later
        pass
    else:
        print("The dataset is empty!")

    eval_dataloader = DataLoader(evaluation_data, batch_size=args.batch_size, shuffle=True)

    # train
    model = BertForClassification()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    train(args.epochs, train_dataloader, model, tokenizer, args.max_length, args.device)

    # test
    for head in range(model.num_heads):
        output_path = "../../output/bert_classification_head_wise/"
        model = torch.load(output_path+f"bert_classification_head_{head}.bin")
        test(head, eval_dataloader, model, tokenizer, args.max_length, args.device)


if __name__ == "__main__":
    main()
