import copy
import logging
import os

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model import DEFAULT_MODEL_NAMES

from config import LoggerConfig

DEFALT_DATASETS = {"xnli": "xnli", "pawsx":"paws-x", "wikiann": "wikiann", "ud": "universal_dependencies"}
DEFALT_LANGUAGES = {
    "xnli": ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh', 'all_languages'],
    "pawsx": ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh', "all_languages"],
    "wikiann": ['ace', 'af', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast', 'ay', 'az', 'ba', 'bar', 'bat-smg', 'be', 'be-x-old', 'bg', 'bh', 'bn', 'bo', 'br', 'bs', 'ca', 'cbk-zam', 'cdo', 'ce', 'ceb', 'ckb', 'co', 'crh', 'cs', 'csb', 'cv', 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'fi', 'fiu-vro', 'fo', 'fr', 'frr', 'fur', 'fy', 'ga', 'gan', 'gd', 'gl', 'gn', 'gu', 'hak', 'he', 'hi', 'hr', 'hsb', 'hu', 'hy', 'ia', 'id', 'ig', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ksh', 'ku', 'ky', 'la', 'lb', 'li', 'lij', 'lmo', 'ln', 'lt', 'lv', 'map-bms', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'mwl', 'my', 'mzn', 'nap', 'nds', 'ne', 'nl', 'nn', 'no', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 'simple', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zea', 'zh', 'zh-classical', 'zh-min-nan', 'zh-yue'],
    "ud": ['af_afribooms', 'akk_pisandub', 'akk_riao', 'aqz_tudet', 'sq_tsa', 'am_att', 'grc_perseus', 'grc_proiel', 'apu_ufpa', 'ar_nyuad', 'ar_padt', 'ar_pud', 'hy_armtdp', 'aii_as', 'bm_crb', 'eu_bdt', 'be_hse', 'bho_bhtb', 'br_keb', 'bg_btb', 'bxr_bdt', 'yue_hk', 'ca_ancora', 'zh_cfl', 'zh_gsd', 'zh_gsdsimp', 'zh_hk', 'zh_pud', 'ckt_hse', 'lzh_kyoto', 'cop_scriptorium', 'hr_set', 'cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt', 'cs_pud', 'da_ddt', 'nl_alpino', 'nl_lassysmall', 'en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pronouns', 'en_pud', 'myv_jr', 'et_edt', 'et_ewt', 'fo_farpahc', 'fo_oft', 'fi_ftb', 'fi_ood', 'fi_pud', 'fi_tdt', 'fr_fqb', 'fr_ftb', 'fr_gsd', 'fr_partut', 'fr_pud', 'fr_sequoia', 'fr_spoken', 'gl_ctg', 'gl_treegal', 'de_gsd', 'de_hdt', 'de_lit', 'de_pud', 'got_proiel', 'el_gdt', 'he_htb', 'qhe_hiencs', 'hi_hdtb', 'hi_pud', 'hu_szeged', 'is_icepahc', 'is_pud', 'id_csui', 'id_gsd', 'id_pud', 'ga_idt', 'it_isdt', 'it_partut', 'it_postwita', 'it_pud', 'it_twittiro', 'it_vit', 'ja_bccwj', 'ja_gsd', 'ja_modern', 'ja_pud', 'krl_kkpp', 'kk_ktb', 'kfm_aha', 'koi_uh', 'kpv_ikdp', 'kpv_lattice', 'ko_gsd', 'ko_kaist', 'ko_pud', 'kmr_mg', 'la_ittb', 'la_llct', 'la_perseus', 'la_proiel', 'lv_lvtb', 'lt_alksnis', 'lt_hse', 'olo_kkpp', 'mt_mudt', 'gv_cadhan', 'mr_ufal', 'gun_dooley', 'gun_thomas', 'mdf_jr', 'myu_tudet', 'pcm_nsc', 'nyq_aha', 'sme_giella', 'no_bokmaal', 'no_nynorsk', 'no_nynorsklia', 'cu_proiel', 'fro_srcmf', 'orv_rnc', 'orv_torot', 'otk_tonqq', 'fa_perdt', 'fa_seraji', 'pl_lfg', 'pl_pdb', 'pl_pud', 'pt_bosque', 'pt_gsd', 'pt_pud', 'ro_nonstandard', 'ro_rrt', 'ro_simonero', 'ru_gsd', 'ru_pud', 'ru_syntagrus', 'ru_taiga', 'sa_ufal', 'sa_vedic', 'gd_arcosg', 'sr_set', 'sms_giellagas', 'sk_snk', 'sl_ssj', 'sl_sst', 'soj_aha', 'ajp_madar', 'es_ancora', 'es_gsd', 'es_pud', 'swl_sslc', 'sv_lines', 'sv_pud', 'sv_talbanken', 'gsw_uzh', 'tl_trg', 'tl_ugnayan', 'ta_mwtt', 'ta_ttb', 'te_mtg', 'th_pud', 'tpn_tudet', 'qtd_sagt', 'tr_boun', 'tr_gb', 'tr_imst', 'tr_pud', 'uk_iu', 'hsb_ufal', 'ur_udtb', 'ug_udt', 'vi_vtb', 'wbp_ufal', 'cy_ccg', 'wo_wtb', 'yo_ytb'],
}
DEFALT_NUM_LABELS = {"xnli": 3, "pawsx":2, "wikiann": 3, "ud": {"POS": 18, "DEP": 3}} # xnli: 0->entailment, 1->neutral, 2->contradiction
TAG_DICT_WIKIANN = {"0": "0", "1": "B-PER", "2": "I-PER", "3": "B-ORG", "4": "I-ORG", "5": "B-LOC", "6": "I-LOC"}
TAG_DICT = {"PER": [1, 2], "ORG": [3, 4], "LOC": [5, 6]}
NER_TAG = ["PER", "LOC", "ORG"]
DEP_TAG_CORE = ["nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp"] # six core relation
DEP_TAG_MODI = ["nmod", "amod", "advmod", "nummod"] # four modifier relation

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

class DataConfig:
    def __init__(self, args):
        self.corpus = DEFALT_DATASETS[args.corpus]
        self.task = args.task
        self.lang = args.lang # or subset
        self.tag_class = args.tag_class
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.src = args.src
        if args.model_config == "M-BERT":
            self.model_config = "mbert"
        elif args.model_config == "XLM-R":
            self.model_config = "xlm"
        self.sample_config = LoggerConfig()


class EvaluationProbing(DataConfig):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger("EvalDataloader")
        self.logger.info(vars(args))
        # checkpoint path and tokenizer
        if args.checkpoints != "NA":
            self.checkpoint = os.path.join(self.sample_config.checkpoints, f"finetune-{self.src}-{self.model_config}")  # /home/weicheng/data_interns/yuansui/models/finetune-pawsx-mbert
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.checkpoint)  # load tokenizer from json
        else:
            self.checkpoint = DEFAULT_MODEL_NAMES[f"{args.model_config}"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        self.pad_token_id = -100

        if self.corpus == "wikiann":
            if self.lang != "all_languages":
                self.dataset = self.load_dataset_wikiann(self.lang, None)
            else:
                self.dataset = self.add_dataset(args.corpus)
        elif self.corpus == "universal_dependencies":
            self.dataset = load_dataset(self.corpus, self.lang) # here self.lang == self.subsets
            if self.dataset.get("train", "") == "":
                # split test set
                df_list = {
                    "train": [load_dataset(self.corpus, self.lang, split='test[:70%]')],
                    "validation": [load_dataset(self.corpus, self.lang, split='test[-30%:]')],
                    "test": [load_dataset(self.corpus, self.lang, split='test[-10%:]')]
                }
                self.dataset = DatasetDict({
                        "train": concatenate_datasets(df_list["train"]),
                        "validation": concatenate_datasets(df_list["validation"]),
                        "test": concatenate_datasets(df_list["test"])
                })
        else:
            if self.lang != "all_languages":
                self.dataset = self.load_dataset_wikiann(self.lang, None)
            else:
                self.dataset = self.add_dataset(args.corpus)

    def add_dataset(self, key):
        df_list = {"train":[], "validation":[], "test":[]}
        if self.corpus == "wikiann":
            for set in ['train', 'validation', 'test']:
                for lang in DEFALT_LANGUAGES[key][:-1]:
                    df_samples = self.load_dataset_wikiann(lang, set)
                    df_list[set].append(df_samples)
        else:
            for set in ['train', 'validation', 'test']:
                for lang in DEFALT_LANGUAGES[key][:-1]:
                    df = load_dataset(self.corpus, lang, split=set)
                    df_samples = df.shuffle(1234)
                    df_list[set].append(df_samples)
        return DatasetDict(
            {
                "train": concatenate_datasets(df_list["train"]),
                "validation": concatenate_datasets(df_list["validation"]),
                "test": concatenate_datasets(df_list["test"])
            }
        )

    def load_dataset_wikiann(self, lang, split):
        """tag_class split"""
        wikiann_dataset = load_dataset(self.corpus, lang, split=split).shuffle(1234)
        for set in ['train', 'validation', 'test']:
            wikiann_dataset[set] = wikiann_dataset[set].filter(
                lambda x: "".join(x['spans']).__contains__(self.tag_class)
            )
        return DatasetDict(
            {
                "train": wikiann_dataset["train"],
                "validation": wikiann_dataset["validation"],
                "test": wikiann_dataset["test"]
            }
        )

    def collote_xnli(self, batch_samples):
        """
        xnli, premise, hypothesis, label
        """
        batch_premise, batch_hypothesis = [], []
        batch_label = []

        for sample in batch_samples:
            batch_premise.append(sample["premise"])
            batch_hypothesis.append(sample["hypothesis"])
            batch_label.append(sample["label"])
        return self.tokenizer(batch_premise, batch_hypothesis, padding="max_length", max_length=self.max_length,
                              truncation=True, return_tensors="pt"), \
               torch.tensor(batch_label)

    def collote_paswx(self, batch_samples):
        """
        pasw-x
        """
        batch_sentence_1, batch_sentence_2 = [], []
        batch_label = []
        for sample in batch_samples:
            batch_sentence_1.append(sample["sentence1"])
            batch_sentence_2.append(sample["sentence2"])
            batch_label.append(sample["label"])
        return self.tokenizer(
            batch_sentence_1,
            batch_sentence_2,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ), torch.tensor(batch_label)

    def collote_wikiann(self, batch_samples):
        """
        wikiann, ner task
        """
        tokens, ner_tags, align_ner_tags = [], [], []
        for sample in batch_samples:
            tokens.append(sample["tokens"])
            ner_tags.append(sample["ner_tags"])
        # drop out other features
        for _ in range(len(ner_tags)):
            for index, value in enumerate(ner_tags[_]):
                if value not in TAG_DICT[self.tag_class]:
                    ner_tags[_][index] = 0
                else:
                    # convert label to 0, 1, 2 (3->1, 4->2; 5->1, 6->2)
                    ner_tags[_][index] = 1 if value in [TAG_DICT[self.tag_class][0]] else 2
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        for i, tag in enumerate(ner_tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            tag_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    tag_ids.append(self.pad_token_id)
                elif word_idx != previous_word_idx:
                    tag_ids.append(tag[word_idx])
                else:
                    tag_ids.append(self.pad_token_id)
                previous_word_idx = word_idx
            align_ner_tags.append(tag_ids)
        return tokenized_inputs, torch.tensor(align_ner_tags)

    def collote_ud_pos(self, batch_samples):
        """
        UD dataset for pos-tagging task
        """
        tokens, pos_tags, align_pos_tags = [], [], []
        for sample in batch_samples:
            tokens.append(sample["lemmas"])
            pos_tags.append(sample["upos"])
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        for i, tag in enumerate(pos_tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            tag_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    tag_ids.append(self.pad_token_id)
                else:
                    tag_ids.append(tag[word_idx])
            align_pos_tags.append(tag_ids)
        return tokenized_inputs, torch.tensor(align_pos_tags)

    def collote_ud_dep(self, batch_samples):
        tokens, head_pair, dep_rel, dep_tag, align_dep_tag = [], [], [], [], []
        for sample in batch_samples:
            tokens.append(sample["lemmas"])
            head_pair.append(sample["head"])
            dep_rel.append(sample["deprel"])
        # # append head annotation
        # for i in range(len(tokens)):
        #     tokens[i].append(f"[SEP]{head}")

        # get the valid_idx
        valid_b_idx = []
        for _ in range(len(head_pair)):
            valid_idx = []
            for dep_idx, value in enumerate(dep_rel[_]):
                if value.__contains__(self.tag_class):
                    valid_idx.append(dep_idx)
            valid_b_idx.append(valid_idx)

        # drop out other features
        dep_tag = copy.deepcopy(head_pair) # deep copy
        for _ in range(len(head_pair)):
            for head_idx, value in enumerate(head_pair[_]):
                dep_tag[_][head_idx] = 0 # nonsense (drop-off)

            if valid_b_idx[_]:
                for val in valid_b_idx[_]:
                    for head_idx, value in enumerate(head_pair[_]):
                        if val == head_idx:
                            dep_tag[_][val] = 2 # dependent
                            if value != 0:
                                dep_tag[_][int(value)-1] = 1 # head

        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        for i, tag in enumerate(dep_tag):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            tag_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    tag_ids.append(self.pad_token_id)
                else:
                    tag_ids.append(tag[word_idx])
            align_dep_tag.append(tag_ids)

        return tokenized_inputs, torch.tensor(align_dep_tag)


    def collote_fn(self, fn):
        collate_fns = {
            "xnli": self.collote_xnli,
            "paws-x": self.collote_paswx,
            "wikiann": self.collote_wikiann,
            "universal_dependencies": self.collote_ud_pos if self.task =="POS" else self.collote_ud_dep
        }
        return collate_fns.get(fn)


    def get_dataloader(self):
        """
        Get dataset = {train:, vali:, test} through this method
        """
        dataloader = {}
        collote_fn = self.collote_fn(self.corpus)
        for set in ['train', 'validation', 'test']:
            dataloader[set] = DataLoader(
                self.dataset[set],
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collote_fn
            )
        self.logger.info(dataloader)
        batch_corpus, batch_label = next(iter(dataloader["train"]))
        self.logger.info(f"tokenizer_pad_token_id: {self.pad_token_id}")
        batch_corpus_shape = {k: v.shape for k, v in batch_corpus.items()}
        self.logger.info(f"batch_corpus_shape: {batch_corpus_shape}")
        self.logger.info(f"batch_label_shape: {batch_label.shape}")

        return dataloader

def main():
    pass

if __name__ == "__main__":
    main()