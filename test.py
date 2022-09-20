# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
#
# class Test(nn.Module):
#
#     def __init__(self, ptm="bert-base-uncased", embed_size=32, num_labels=2):
#         super(Test, self).__init__()
#         self.backbone = AutoModel.from_pretrained(ptm)
#         self.num_heads = self.backbone.config.num_attention_heads
#         self.num_labels = num_labels
#         for p in self.backbone.parameters():
#             p.requires_grad = False  # freeze the backbone model
#         self.classifier = nn.Sequential(
#             nn.Linear(64, embed_size),
#             nn.Dropout(0.5),
#             nn.Linear(embed_size, num_labels)
#         )
#
#     def forward(self, input_ids, attention_mask):
#         backbone = self.backbone(input_ids=input_ids,
#                                  attention_mask=attention_mask,
#                                  output_hidden_states=True)  # output all the hidden states rather than the last layer
#         layers = list(backbone.hidden_states)
#         heads = []
#         for layer_idx in range(len(layers)):
#             for head_idx in range(self.num_heads):
#                 split_heads = layers[layer_idx].chunk(self.num_heads, 2) # split 768 into 12 sections (heads)
#                 heads.append(self.classifier(split_heads[head_idx]))
#         return heads
#
# model = Test()
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# inputs = tokenizer(["Someone like you xadsdw who you are", "In other European markets, share prices closed sharply higher in Frankfurt and Zurich and posted moderate rises in Stockholm"], return_tensors="pt", padding=True, truncation=True)
# outputs = model(inputs["input_ids"], inputs["attention_mask"])

# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# final_score = np.random.rand(10,12)
# sns_fig = sns.heatmap(final_score, vmin=0, vmax=1, cmap="YlGnBu")
# plt.show()
# plt.clf()
# sns.barplot(x=[1,2,3,4],y=np.random.rand(4),)
# sns.color_palette("light:#5A9", as_cmap=True)
# plt.show()
#
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# bar = sns.histplot(data=data, x='Q1',color='#42b7bd')
# # you can search color picker in google, and get hex values of you fav color
#
# patch_h = [patch.get_height() for patch in bar.patches]
# # patch_h contains the heights of all the patches now
#
# idx_tallest = np.argmax(patch_h)
# # np.argmax return the index of largest value of the list
#
# bar.patches[idx_tallest].set_facecolor('#a834a8')
#
# #this will do the trick.


import seaborn
import numpy as np
import matplotlib.pyplot as plt

values = np.array([2,5,3,6,4,7,1])
idx = np.array(list('abcdefg'))

ax = seaborn.histplot(x=idx, y=values, color='#42b7bd') # or use ax=your_axis_object

patch_h = [patch.get_height() for patch in ax.patches]
# patch_h contains the heights of all the patches now

idx_tallest = np.argmax(patch_h)
ax.patches[idx_tallest].set_facecolor('#a834a8')
plt.show()



