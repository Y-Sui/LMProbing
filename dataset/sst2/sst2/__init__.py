import pandas as pd
from torch.utils.data import Dataset, DataLoader

_SST = "../../dataset/sst2/data/"
_SSTs = {
    "train": _SST + "train.tsv",
    "valid": _SST + "valid.tsv",
    "test.py": _SST + "test.py.tsv",
}

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__

class SST2(Dataset):

    def __init__(self,source=""):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = pd.read_csv(_SSTs[source], sep='\t').to_numpy()
        self.n_samples = xy.shape[0]

        # here the first column is the text, the second column is the label
        self.x_data = xy[:, 0]  # size [n_samples, n_features]
        labels = xy[:, 1]  # size [n_samples, 1]
        self.y_data = []
        for i in range(len(labels)):
            self.y_data.append(0 if labels[i] == "negative" else 1)


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
