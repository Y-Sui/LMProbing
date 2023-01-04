# library
import argparse
import os.path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer


def plot_head_magnitude(layers, heads, features, args):

    # Create long format
    # people = np.repeat(("A", "B", "C", "D", "E"), 5)
    # feature = list(range(1, 6)) * 5
    # value = np.random.random(25)
    df = pd.DataFrame({'layers': layers, 'heads': heads, 'features': features})


    # Turn long format into a wide format
    df_wide = df.pivot_table(index='layers', columns='heads', values='features')

    # Sort by rows (layers)
    df_sort = df_wide.sort_index(ascending=False)

    sns.set(font_scale=2.5)
    # plot it
    plt.figure(figsize=(10, 9))
    plt.rc('font', size=18)

    if args.normalize is False:
        if args.plot_difference is False:
            # argument plotting
            df_sort[df_sort[df_sort > -0.0002] < 0.0002] = df_sort[df_sort[df_sort > -0.0002] < 0.0002] * 0.3  # -0.0005 ~ 0.0005
            df_sort[df_sort[df_sort > 0.0002] < 0.0005] = df_sort[df_sort[df_sort > 0.0002] < 0.0005] * 0.5  # 0.0005 ~ 0.0010
            df_sort[df_sort[df_sort < -0.0002] > -0.0005] = df_sort[df_sort[df_sort < -0.0002] > -0.0005] * 1.6  # -0.0005 ~ -0.0010
            df_sort[df_sort < -0.0005] = df_sort[df_sort < -0.0005] * 1.8
            df_sort[df_sort > 0.0005] = df_sort[df_sort > 0.0005] * 1.8
        else:
            # argument plotting
            df_sort[df_sort[df_sort > -0.5] < 0.5] = df_sort[df_sort[df_sort > -0.5] < 0.5] * 0.3
            df_sort[df_sort[df_sort > 0.5] < 1.0] = df_sort[df_sort[df_sort > 0.5] < 1.0] * 0.6
            df_sort[df_sort[df_sort < -0.5] > -1.0] = df_sort[df_sort[df_sort < -0.5] > -1.0] * 0.6
            df_sort[df_sort > 1.0] = df_sort[df_sort > 1.0] * 1.5
            df_sort[df_sort < 1.0] = df_sort[df_sort < 1.0] * 1.5
    else:
        # argument plotting
        df_sort[df_sort < 0.3] = df_sort[df_sort < 0.3] * 0.3
        df_sort[df_sort[df_sort > 0.3] < 0.5] = df_sort[df_sort[df_sort > 0.3] < 0.5] * 0.5
        df_sort[df_sort > 0.5] = df_sort[df_sort > 0.5] * 1.6

    if args.plot_difference:
        sns.heatmap(df_sort, cbar=False, center=0)
    else:
        sns.heatmap(df_sort, cbar=False)

    # save the plotting
    if args.plot_difference:
        if args.normalize:
            plt.savefig(os.path.join("./magnitudes/", f"{args.checkpoints.split('/')[-1]}_difference.pdf"))
        else:
            if args.checkpoints.__contains__("//home/weicheng") is False and args.checkpoints2 is not None:
                plt.savefig(os.path.join("./magnitudes/fpd", f"{args.checkpoints2.split('/')[-1]}_unnormalized_difference_raw.pdf"))
            else:
                plt.savefig(os.path.join("./magnitudes/", f"{args.checkpoints2.split('/')[-1]}_unnormalized_difference.pdf"))
    else:
        if args.normalize:
            plt.savefig(os.path.join("./magnitudes/", f"{args.checkpoints.split('/')[-1]}.pdf"))
        else:
            plt.savefig(os.path.join("./magnitudes/", f"{args.checkpoints.split('/')[-1]}_unnormalized.pdf"))
    plt.clf()

def load_model_checkpoints(args):
    model = AutoModel.from_pretrained(args.checkpoints)
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
            if args.normalize:
                features.append((np.var(head_weights) - np.min(head_weights)) / (np.max(head_weights) - np.min(head_weights)))  # normalize
            else:
                features.append(np.mean(head_weights))
    features = np.array(features)

    # plot
    plot_head_magnitude(layers, heads, features, args)

def obtain_KQV(model_base):
    component = list(model_base.state_dict())  # convert OrderedDict to list
    Q, K, V = [], [], []
    for param_tensor in component:
        if param_tensor.__contains__("attention"):
            if param_tensor.__contains__("query.weight"):
                Q.append(model_base.state_dict()[param_tensor].chunk(12, 1))  # split 768 into 12 heads
            elif param_tensor.__contains__("key.weight"):
                K.append(model_base.state_dict()[param_tensor].chunk(12, 1))
            elif param_tensor.__contains__("value.weight"):
                V.append(model_base.state_dict()[param_tensor].chunk(12, 1))
    return Q, K, V

def plot_difference_checkpoints(args):
    model_base = AutoModel.from_pretrained(args.checkpoints)
    model_bonus = AutoModel.from_pretrained(args.checkpoints2)
    Q_base, K_base, V_base = obtain_KQV(model_base)
    Q_bonus, K_bonus, V_bonus = obtain_KQV(model_bonus)
    # layers = [f"l{i+1}" for i in range(12)] * 12
    layers = []
    for i in range(12):
        for j in range(12):
            layers.append(i + 1)
    heads = [i + 1 for i in range(12)] * 12
    features = []
    for i in range(len(Q_bonus)):
        for j in range(len(Q_bonus[0])):
            head_weights = (K_bonus[i][j] + Q_bonus[i][j] + V_bonus[i][j]).numpy() - (K_base[i][j] + Q_base[i][j] + V_base[i][j]).numpy()
            if args.normalize:
                features.append((np.var(head_weights) - np.min(head_weights)) / (np.max(head_weights) - np.min(head_weights)))  # normalize
            else:
                features.append(np.mean(head_weights))
    features = np.array(features)

    # plot
    plot_head_magnitude(layers, heads, features, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, default="xlm-roberta-base")
    parser.add_argument("--checkpoints2", type=str, default="xlm-roberta-large")
    parser.add_argument("--normalize", action="store_true", help="whether to normalize the output", default=False)
    parser.add_argument("--plot_difference", action="store_true", help="whether to plot difference", default=False)
    args = parser.parse_args()
    if args.plot_difference:
        plot_difference_checkpoints(args)
    else:
        load_model_checkpoints(args)

if __name__ == "__main__":
    main()





