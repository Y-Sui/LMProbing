import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Test(nn.Module):

    def __init__(self, ptm="bert-base-uncased", embed_size=256, num_labels=2):
        super(Test, self).__init__()
        self.backbone = AutoModel.from_pretrained(ptm)
        self.num_heads = self.backbone.config.num_attention_heads
        self.num_labels = num_labels
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze the backbone model
        self.last_hidden_state_size = self._get_last_hidden_state_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.last_hidden_state_size, embed_size),
            nn.Dropout(0.5),
            nn.Linear(embed_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)  # output all the hidden states rather than the last layer
        layers = list(backbone.hidden_states)
        heads = torch.rand(13*self.num_heads, attention_mask.shape[0], attention_mask.shape[1], self.num_labels) # save each layer's output
        for layer_idx in range(len(layers)):
            for head_idx in range(self.num_heads):
                x = layers[layer_idx][:, head_idx, :]
                if layer_idx == 0 and head_idx != 0:
                    split_head = layers[layer_idx][:, head_idx, :]
                    heads[1 * head_idx] = self.classifier(split_head)
                elif layer_idx != 0 and head_idx == 0:
                    split_head = layers[layer_idx][:, head_idx, :]
                    heads[layer_idx * 1] = self.classifier(split_head)
                else:
                    split_head = layers[layer_idx][:, head_idx, :]
                    y = self.classifier(split_head)
                    heads[layer_idx * head_idx] = self.classifier(split_head)

        return heads

    def _get_last_hidden_state_size(self):
        backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states
        return backbone.hidden_states[0].shape[-1]

model = Test()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Someone like you xadsdw who you are", return_tensors="pt")
outputs = model(inputs["input_ids"], inputs["attention_mask"])
print(outputs)
