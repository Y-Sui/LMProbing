import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset.sst2.sst2 import SST2


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


device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("../../module/bert-base-uncased")
model = BertForClassification()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()))  # only update the parameters who are set to requires_grad

epochs = 1
training_data = SST2()
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

for epoch in range(epochs):
    for batch in train_dataloader:
        data = list(batch[0])
        targets = torch.tensor(list(batch[1]))

        optimizer.zero_grad()
        encoding = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True, max_length=50,
                                               add_special_tokens=True)

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        outputs = outputs[:, -1]
        outputs = torch.log_softmax(outputs, dim=1)

        loss = criterion(outputs, targets)
        print(loss)
        loss.backward()
        optimizer.step()

# save the model
PATH = "../../output/bert_classification"
torch.save(model.state_dict(), PATH+".bin")

# test
model.load_state_dict(torch.load(PATH+".bin"))
input = "Pretty much sucks , but has a funny moment or two"
model_input = tokenizer(input, padding="max_length", max_length=50, truncation=True, return_tensors="pt")

output = model(model_input["input_ids"], model_input["attention_mask"])

print(torch.log_softmax(output[:,-1], dim=1))
