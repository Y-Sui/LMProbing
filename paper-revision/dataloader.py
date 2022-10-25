import logging

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

DEFALT_DATASETS = {"xnli": "xnli", "pawsx":"paws-x", "wikiann": "wikiann", "ud": "universal_dependencies"}
DEFALT_LANGUAGES = {
    "xnli": ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh', 'all_languages'],
    "pawsx": ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh'],
    "wikiann": ['ace', 'af', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast', 'ay', 'az', 'ba', 'bar', 'bat-smg', 'be', 'be-x-old', 'bg', 'bh', 'bn', 'bo', 'br', 'bs', 'ca', 'cbk-zam', 'cdo', 'ce', 'ceb', 'ckb', 'co', 'crh', 'cs', 'csb', 'cv', 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'fi', 'fiu-vro', 'fo', 'fr', 'frr', 'fur', 'fy', 'ga', 'gan', 'gd', 'gl', 'gn', 'gu', 'hak', 'he', 'hi', 'hr', 'hsb', 'hu', 'hy', 'ia', 'id', 'ig', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ksh', 'ku', 'ky', 'la', 'lb', 'li', 'lij', 'lmo', 'ln', 'lt', 'lv', 'map-bms', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'mwl', 'my', 'mzn', 'nap', 'nds', 'ne', 'nl', 'nn', 'no', 'nov', 'oc', 'or', 'os', 'pa', 'pdc', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'rw', 'sa', 'sah', 'scn', 'sco', 'sd', 'sh', 'si', 'simple', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'szl', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xmf', 'yi', 'yo', 'zea', 'zh', 'zh-classical', 'zh-min-nan', 'zh-yue'],
    "ud": ['af_afribooms', 'akk_pisandub', 'akk_riao', 'aqz_tudet', 'sq_tsa', 'am_att', 'grc_perseus', 'grc_proiel', 'apu_ufpa', 'ar_nyuad', 'ar_padt', 'ar_pud', 'hy_armtdp', 'aii_as', 'bm_crb', 'eu_bdt', 'be_hse', 'bho_bhtb', 'br_keb', 'bg_btb', 'bxr_bdt', 'yue_hk', 'ca_ancora', 'zh_cfl', 'zh_gsd', 'zh_gsdsimp', 'zh_hk', 'zh_pud', 'ckt_hse', 'lzh_kyoto', 'cop_scriptorium', 'hr_set', 'cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt', 'cs_pud', 'da_ddt', 'nl_alpino', 'nl_lassysmall', 'en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut', 'en_pronouns', 'en_pud', 'myv_jr', 'et_edt', 'et_ewt', 'fo_farpahc', 'fo_oft', 'fi_ftb', 'fi_ood', 'fi_pud', 'fi_tdt', 'fr_fqb', 'fr_ftb', 'fr_gsd', 'fr_partut', 'fr_pud', 'fr_sequoia', 'fr_spoken', 'gl_ctg', 'gl_treegal', 'de_gsd', 'de_hdt', 'de_lit', 'de_pud', 'got_proiel', 'el_gdt', 'he_htb', 'qhe_hiencs', 'hi_hdtb', 'hi_pud', 'hu_szeged', 'is_icepahc', 'is_pud', 'id_csui', 'id_gsd', 'id_pud', 'ga_idt', 'it_isdt', 'it_partut', 'it_postwita', 'it_pud', 'it_twittiro', 'it_vit', 'ja_bccwj', 'ja_gsd', 'ja_modern', 'ja_pud', 'krl_kkpp', 'kk_ktb', 'kfm_aha', 'koi_uh', 'kpv_ikdp', 'kpv_lattice', 'ko_gsd', 'ko_kaist', 'ko_pud', 'kmr_mg', 'la_ittb', 'la_llct', 'la_perseus', 'la_proiel', 'lv_lvtb', 'lt_alksnis', 'lt_hse', 'olo_kkpp', 'mt_mudt', 'gv_cadhan', 'mr_ufal', 'gun_dooley', 'gun_thomas', 'mdf_jr', 'myu_tudet', 'pcm_nsc', 'nyq_aha', 'sme_giella', 'no_bokmaal', 'no_nynorsk', 'no_nynorsklia', 'cu_proiel', 'fro_srcmf', 'orv_rnc', 'orv_torot', 'otk_tonqq', 'fa_perdt', 'fa_seraji', 'pl_lfg', 'pl_pdb', 'pl_pud', 'pt_bosque', 'pt_gsd', 'pt_pud', 'ro_nonstandard', 'ro_rrt', 'ro_simonero', 'ru_gsd', 'ru_pud', 'ru_syntagrus', 'ru_taiga', 'sa_ufal', 'sa_vedic', 'gd_arcosg', 'sr_set', 'sms_giellagas', 'sk_snk', 'sl_ssj', 'sl_sst', 'soj_aha', 'ajp_madar', 'es_ancora', 'es_gsd', 'es_pud', 'swl_sslc', 'sv_lines', 'sv_pud', 'sv_talbanken', 'gsw_uzh', 'tl_trg', 'tl_ugnayan', 'ta_mwtt', 'ta_ttb', 'te_mtg', 'th_pud', 'tpn_tudet', 'qtd_sagt', 'tr_boun', 'tr_gb', 'tr_imst', 'tr_pud', 'uk_iu', 'hsb_ufal', 'ur_udtb', 'ug_udt', 'vi_vtb', 'wbp_ufal', 'cy_ccg', 'wo_wtb', 'yo_ytb'],
}
DEFALT_NUM_LABELS = {"xnli": 2, "pawsx":2, "wikiann": 7, "ud": 5}
TAG_DICT_WIKIANN = {"O": 0, "1": "B-PER", "2": "I-PER", "3": "B-ORG", "4": "I-ORG", "5": "B-LOC", "6": "I-LOC"}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

class DataConfig:
    def __init__(self, args):
        self.corpus = DEFALT_DATASETS[args.corpus]
        self.lang = args.lang # or subset
        self.tokenizer_config = args.tokenizer_config
        self.max_length = args.max_length
        self.batch_size = args.batch_size

class EvaluationProbing(DataConfig):
    def __init__(self, args):
        super().__init__(args)
        self.logger = logging.getLogger("EvalDataloader")
        self.logger.info(vars(args))
        self.dataset = load_dataset(self.corpus, self.lang)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_config)

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
        return self.tokenizer(batch_sentence_1, batch_sentence_2, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt"), \
               torch.tensor(batch_label)

    def collote_wikiann(self, batch_samples):
        """
        wikiann, ner task
        """
        tokens, ner_tags = [], []
        for sample in batch_samples:
            tokens.append(sample["tokens"])
            ner_tags.append(sample["ner_tags"])
        return self.tokenizer(tokens, is_split_into_words=True, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt"), \
               torch.tensor(ner_tags)

    def collote_ud(self, batch_samples):
        pass

    def collote_fn(self, fn):
        collate_fns = {
            "xnli": self.collote_xnli,
            "paws-x": self.collote_paswx,
            "wikiann": self.collote_wikiann,
            "universal_dependencies": self.collote_ud
        }
        return collate_fns.get(fn)

    def get_dataloader(self):
        """
        Get dataset = {train:, vali:, test} through this method
        """
        dataloader = {}
        collote_fn = self.collote_fn(self.corpus)
        for set in ['train', 'validation', 'test']:
            dataloader[set] = DataLoader(self.dataset[set], batch_size=self.batch_size, shuffle=True, collate_fn=collote_fn)
        self.logger.info(dataloader)
        batch_corpus, batch_label = next(iter(dataloader["train"]))
        batch_corpus_shape = {k: v.shape for k, v in batch_corpus.items()}
        self.logger.info(f"batch_corpus_shape: {batch_corpus_shape}")
        self.logger.info(f"batch_label_shape: {batch_label.shape}")

        return dataloader

def main():
    pass

if __name__ == "__main__":
    main()