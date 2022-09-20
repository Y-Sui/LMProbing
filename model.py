import torch
import torch.nn as nn
from transformers import AutoModel


class Bert_4_Classification_Layer_Wise(nn.Module):

    def __init__(self, ptm="bert-base-uncased", embed_size=256, num_labels=2):
        super(Bert_4_Classification_Layer_Wise, self).__init__()
        self.backbone = AutoModel.from_pretrained(ptm)
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
        layers = list(backbone.hidden_states)  # save each layer's output
        for layer_idx in range(len(layers)):
            layers[layer_idx] = self.classifier(layers[layer_idx])
        return layers

    def _get_last_hidden_state_size(self):
        backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states
        return backbone.hidden_states[0].shape[-1]

class Bert_4_Classification_Head_Wise(nn.Module):
    """
    Split the layer wise
    """

    def __init__(self, ptm="bert-base-uncased", embed_size=32, num_labels=2):
        super(Bert_4_Classification_Head_Wise, self).__init__()
        self.backbone = AutoModel.from_pretrained(ptm)
        self.num_heads = self.backbone.config.num_attention_heads
        self.last_hidden_state_size = self._get_last_hidden_state_size()
        self.num_labels = num_labels
        for p in self.backbone.parameters():
            p.requires_grad = False  # freeze the backbone model
        self.classifier = nn.Sequential(
            nn.Linear(64, embed_size),
            nn.Dropout(0.5),
            nn.Linear(embed_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        backbone = self.backbone(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)  # output all the hidden states rather than the last layer
        layers = list(backbone.hidden_states)
        heads = []
        for layer_idx in range(len(layers)):
            for head_idx in range(self.num_heads):
                split_heads = layers[layer_idx].chunk(self.num_heads, 2) # split 768 into 12 sections (heads)
                heads.append(self.classifier(split_heads[head_idx]))
        return heads

    def _get_last_hidden_state_size(self):
        backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
        self.hidden_states = backbone.hidden_states
        return backbone.hidden_states[0].shape[-1]


# class Bert_4_Classification_Head_Wise(nn.Module):
#
#     def __init__(self, ptm="bert-base-uncased", embed_size=256, num_labels=2):
#         super(Bert_4_Classification_Head_Wise, self).__init__()
#         self.num_labels = num_labels
#         self.backbone = AutoModel.from_pretrained(ptm)
#         self.num_heads = self.backbone.config.num_attention_heads
#         for p in self.backbone.parameters():
#             p.requires_grad = False  # freeze the backbone model
#         self.last_hidden_state_size = self._get_last_hidden_state_size()
#         self.classifier = nn.Sequential(
#             nn.Linear(self.last_hidden_state_size, embed_size),
#             nn.Dropout(0.5),
#             nn.Linear(embed_size, num_labels)
#         )
#
#     def forward(self, input_ids, attention_mask):
#         head_mask = [0 for _ in range(self.num_heads)]
#         head_mask_list = []
#         for i in range(len(head_mask)):
#             head_mask_n = head_mask[:]
#             head_mask_n[i] = 1
#             head_mask_list.append(head_mask_n)
#         head_mask_list = torch.tensor(head_mask_list).to('cuda') # add cuda to make sure the tensors on the same device
#         heads = torch.rand(self.num_heads*13, attention_mask.shape[0], attention_mask.shape[1], self.num_labels)
#         for head_idx in range(len(head_mask_list)):
#             backbone = self.backbone(input_ids=input_ids,
#                                      attention_mask=attention_mask,
#                                      output_hidden_states=True,
#                                      head_mask=head_mask_list[head_idx])  # Set the head_mask
#             layers = list(backbone.hidden_states)  # save each layer's output
#             for layer_idx in range(len(layers)):
#                 heads[head_idx * layer_idx] = self.classifier(layers[layer_idx])
#         return heads
#
#     def _get_last_hidden_state_size(self):
#         backbone = self.backbone(torch.tensor([[1, 1]]), torch.tensor([[1, 1]]), output_hidden_states=True)
#         self.hidden_states = backbone.hidden_states
#         return backbone.hidden_states[0].shape[-1]

