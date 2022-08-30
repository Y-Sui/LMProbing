import os
import json
import random

import pandas as pd


def get_files_path(filePath):
    raw_files = os.listdir(filePath)
    for i in range(len(raw_files)):
        raw_files[i] = filePath + "/" + raw_files[i]
    return raw_files

def concat_files(nerPath="./ner", chunkPath="./chunk"):
    ner_files = get_files_path(nerPath)
    chunk_files = get_files_path(chunkPath)
    return ner_files + chunk_files

def get_pure_tokens(tokenList):
    tokens = []
    for i in range(len(tokenList)):
        tokenList[i] = tokenList[i].\
            replace("O-", "").replace("X-", "").replace("B-LOC-", "").replace("B-MISC-", "").\
            replace("B-ORG-","").replace("B-PER-", "").replace("B-ADJP-","").replace("B-ADVP-", "").\
            replace("B-CONJP-", "").replace("B-INTJ-", "").replace("B-LST-", "").replace("B-NP-", "").\
            replace("B-PP-","").replace("B-PRT-", "").replace("B-SBAR-", "").replace("B-VP-", ""). \
            replace("I-LOC-", "").replace("I-MISC-", "").replace("I-ORG-", "").replace("I-PER-", "").\
            replace("I-ADJP-","").replace("I-ADVP-", "").replace("I-CONJP-", "").replace("I-INTJ-", "").\
            replace("I-LST-", "").replace("I-NP-", "").replace("I-PP-", "").replace("I-PRT-","").\
            replace("I-SBAR-", "").replace("I-VP-", "")
        tokens.append(tokenList[i].split())
    return tokens


def get_tags(tagList, flag="ner"):
    if flag == "ner":
        tag_dict = {"O":0,"B-ORG":1,"I-ORG":2,"B-PER":3,"I-PER":4,"B-MISC":5,"I-MISC":6,"B-LOC":7,"I-LOC":8}
    if flag == "chunk":
        tag_dict = {"O":0,"B-ADJP":1,"I-ADJP":2,"B-ADVP":3,"I-ADVP":4,"B-CONJP":5,"I-CONJP":6,"B-INTJ":7,"I-INTJ":8,
                      "B-LST":9, "I-LST":10, "B-NP":11, "I-NP":12, "B-PP":13, "I-PP":14, "B-PRT":15, "I-PRT":16,
                      "B-SBAR":17, "I-SBAR":18, "B-VP":19, "I-VP":20}
    tag_keys = list(tag_dict.keys())
    for i in range(len(tagList)):
        tagList[i] = tagList[i].split()
        for j in range(len(tagList[i])):
            for index in range(len(tag_keys)):
                if tagList[i][j].__contains__(f"{tag_keys[index]+'-'}"):
                    tagList[i][j] = tag_dict[f"{tag_keys[index]}"]
                    break
    # how to deal with X
    for i in range(len(tagList)):
        for j in range(len(tagList[i])):
            if isinstance(tagList[i][j], str) and tagList[i][j].__contains__("X-"):
                tagList[i][j] = tagList[i][j-1]
    return tagList

def data_split(full_list, ratio, shuffle=False):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


# raw_files = concat_files("./ner", "./chunk")
for n in ["ner", "chunk"]:
    raw_files = get_files_path(f"./{n}") # obtain the ner files
    random_files = []
    for file in range(len(raw_files)):
        pd_reader = pd.read_csv(raw_files[file], header=None)[1].tolist()
        token_list = pd_reader.copy()
        tag_list = pd_reader.copy()
        tokens = get_pure_tokens(token_list) # get the pure tokens
        flag = "ner" if "ner" in raw_files[file] else "chunk"
        tags = get_tags(tag_list, flag) # get the tags

        # save json file
        save_file = []
        for i in range(len(tokens)):
            save_file.append({'tokens': tokens[i], f"{flag}_tags": tags[i]})
        random_files.extend(save_file)
        # save samples for each labels without data_split
        # json_file = raw_files[file].replace(".csv", ".json")
        # with open(json_file, 'w') as fp:
        #     json.dump(save_file, fp)

    train_files, eval_files = data_split(random_files, 0.7, shuffle=True)
    # train_files = dict(zip([f"id_{i}" for i in range(len(train_files))], train_files)) # convert list to dict, Note that dict(key:value)
    # eval_files = dict(zip([f"id_{j}" for j in range(len(eval_files))], eval_files))
    train_files = {"data": train_files}
    eval_files = {"data": eval_files}
    with open(f"{n}_train.json", "w") as fp:
        json.dump(train_files, fp)
    with open(f"{n}_eval.json", "w") as fp:
        json.dump(eval_files, fp)


