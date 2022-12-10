# library
import argparse
import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer


def plot_head_magnitude(layers, heads, features, checkpoint_path):

    # Create long format
    # people = np.repeat(("A", "B", "C", "D", "E"), 5)
    # feature = list(range(1, 6)) * 5
    # value = np.random.random(25)
    df = pd.DataFrame({'layers': layers, 'heads': heads, 'features': features})


    # Turn long format into a wide format
    df_wide = df.pivot_table(index='layers', columns='heads', values='features')

    # plot it
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_wide, annot=True)

    # save the plotting
    plt.savefig(os.path.join("./magnitudes/", f"{checkpoint_path.split('/')[-1]}.png"))
    plt.clf()

def load_model_checkpoints(checkpoint_path):
    model = AutoModel.from_pretrained(checkpoint_path)
    component = list(model.state_dict()) # convert OrderedDict to list
    Q, K, V = [], [], []
    for param_tensor in component:
        if param_tensor.__contains__("attention"):
            if param_tensor.__contains__("query.weight"):
                Q.append(model.state_dict()[param_tensor].chunk(12,1)) # split 768 into 12 heads
            elif param_tensor.__contains__("key.weight"):
                K.append(model.state_dict()[param_tensor].chunk(12,1))
            elif param_tensor.__contains__("value.weight"):
                V.append(model.state_dict()[param_tensor].chunk(12,1))
    # layers = [f"l{i+1}" for i in range(12)] * 12
    layers = []
    for i in range(12):
        for j in range(12):
            layers.append(i+1)
    heads = [i+1 for i in range(12)] * 12
    features = []
    for i in range(len(Q)):
        for j in range(len(Q[0])):
            head_weights = ((K[i][j] + Q[i][j] + V[i][j]).numpy())
            features.append((np.var(head_weights) - np.min(head_weights)) / (np.max(head_weights) - np.min(head_weights))) # normalize
    features = np.array(features)

    # plot
    plot_head_magnitude(layers, heads, features, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, default="xlm-roberta-base")
    args = parser.parse_args()
    load_model_checkpoints(args.checkpoints)

if __name__ == "__main__":
    main()





