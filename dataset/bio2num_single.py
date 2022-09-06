import os
import json
import random
import re
import os

import pandas as pd


def get_files_path(filePath):
    """
    return the file list
    """
    raw_files = os.listdir(filePath)
    file_paths = []
    for i in range(len(raw_files)):
        if raw_files[i].find("train") == -1 and raw_files[i].find("eval") == -1:
           file_paths.append(filePath + "/" + raw_files[i])
        # else:
        #     raw_files.pop(i)  # remove the elements
    return file_paths


def get_pure_tokens(tokenList, category):
    """
    filter out all the annotated tags to get the pure token list
    """
    filter = ["O-", "X-", f"B-{category}-", f"I-{category}-"]
    if category.islower():
        filter = ["O-", "X-", "B-head-", "B-dependent-"]
    if category == "VP":
        filter = ["O-", "X-", "B-ADVP-", "I-ADVP-", "B-VP-", "I-VP-"]
    if category == "":
        filter = ["O-", "X-", "B-:-", "I-:-"]
    if category == "EX":
        filter = ["O-", "X-", "B-E", "I-E"]
    if category == "TO":
        filter = ["O-", "X-", "B-T", "I-T"]
    tokens = []
    for i in range(len(tokenList)):
        for j in range(len(filter)):
            tokenList[i] = tokenList[i].replace(filter[j], "")
        tokens.append(tokenList[i].split())
    return tokens


def get_tags(tagList, flag="ner"):
    """
    generate the mixed tag list
    """
    if flag == "ner":
        tag_dict = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-PER": 3, "I-PER": 4, "B-MISC": 5, "I-MISC": 6, "B-LOC": 7,
                    "I-LOC": 8}
    if flag == "chunk":
        tag_dict = {"O": 0, "B-ADJP": 1, "I-ADJP": 2, "B-ADVP": 3, "I-ADVP": 4, "B-CONJP": 5, "I-CONJP": 6, "B-INTJ": 7,
                    "I-INTJ": 8, "B-LST": 9, "I-LST": 10, "B-NP": 11, "I-NP": 12, "B-PP": 13, "I-PP": 14, "B-PRT": 15,
                    "I-PRT": 16, "B-SBAR": 17, "I-SBAR": 18, "B-VP": 19, "I-VP": 20}
    tag_keys = list(tag_dict.keys())
    for i in range(len(tagList)):
        tagList[i] = tagList[i].split()
        for j in range(len(tagList[i])):
            for index in range(len(tag_keys)):
                if tagList[i][j].__contains__(f"{tag_keys[index] + '-'}"):
                    tagList[i][j] = tag_dict[f"{tag_keys[index]}"]
                    break
    # how to deal with X
    for i in range(len(tagList)):
        for j in range(len(tagList[i])):
            if isinstance(tagList[i][j], str) and tagList[i][j].__contains__("X-"):
                tagList[i][j] = tagList[i][j - 1]
    return tagList


def get_single_tags(tagList, category="LOC"):
    """
    generate the single tag list
    """
    tag_dict = {"O": 0, f"B-{category}": 1, f"I-{category}": 2}
    if category.islower():
        tag_dict = {"O": 0, "B-head": 1, "B-dependent": 2}
    if category == "VP":
        tag_dict = {"O": 0, "B-ADVP": 1, "I-ADVP": 2, "B-VP":3, "I-VP": 4}
    if category == "":
        tag_dict = {"O": 0, "B-:": 1, "I-:": 2}
    if category == "EX":
        tag_dict = {"O": 0, "B-E:": 1, "I-E:": 2}
    if category == "TO":
        tag_dict = {"O": 0, "B-T:": 1, "I-T:": 2}
    tag_keys = list(tag_dict.keys())
    for i in range(len(tagList)):
        tagList[i] = tagList[i].split()
        for j in range(len(tagList[i])):
            for index in range(len(tag_keys)):
                if str(tagList[i][j]).__contains__(f"{tag_keys[index] + '-'}"):
                    tagList[i][j] = tag_dict[f"{tag_keys[index]}"]
        # break
    # how to deal with X
    for i in range(len(tagList)):
        for j in range(len(tagList[i])):
            if isinstance(tagList[i][j], str) and tagList[i][j].__contains__("X-"):
                tagList[i][j] = tagList[i][j - 1]
    return tagList


def data_split(full_list, ratio, shuffle=False):
    """
    split the samples into training and evaluation
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def main():
    for n in ["ner", "chunk", "pos-fixed", "dependency-fixed"]:
        print(f"Start to generate {n} datasets...")
        raw_files = get_files_path(f"./{n}")  # obtain the {ner} files list
        random_files = []
        for file in range(len(raw_files)):
            pd_reader = pd.read_csv(raw_files[file], header=None)[1].tolist()
            token_list = pd_reader.copy()
            tag_list = pd_reader.copy()
            # flag = "ner" if "ner" in raw_files[file] else "chunk"
            # category = "".join(re.findall(r'[A-Z]', raw_files[file]))  # get the Uppercase file name which refers to the categories of the labels
            category = raw_files[file].split("_")[-1].replace(".csv", "")
            tokens = get_pure_tokens(token_list, category)  # get the pure tokens
            tags = get_single_tags(tag_list, category)  # get the tags

            # save json file
            samples = []
            for i in range(len(tokens)):
                samples.append({'tokens': tokens[i], f"{n}_tags": tags[i]})

            # down-sampling
            train_samples, eval_samples = data_split(samples, 0.7, shuffle=True)
            train_samples = {"data": train_samples}
            eval_samples = {"data": eval_samples}

            # save samples for each labels without data_split
            json_file = raw_files[file].replace(".csv", "")
            with open(f"{json_file}_train.json", 'w') as fp:
                json.dump(train_samples, fp)
            with open(f"{json_file}_eval.json", 'w') as fp:
                json.dump(eval_samples, fp)
            print(f"The {n} dataset corresponding single files have been saved...")
            # save the mixed tags
            random_files.extend(samples)

        train_files, eval_files = data_split(random_files, 0.7, shuffle=True)
        train_files = {"data": train_files}
        eval_files = {"data": eval_files}
        with open(f"{n}_train.json", "w") as fp:
            json.dump(train_files, fp)
        with open(f"{n}_eval.json", "w") as fp:
            json.dump(eval_files, fp)
        print(f"The {n} dataset corresponding mixed files have been saved...")


if __name__ == "__main__":
    main()
