import pandas as pd
from .utils import *
import time
import re
from pathlib import Path
from nltk.corpus import wordnet
from .tokenizer import get_tokenizer
from tqdm import tqdm
import random
from random import shuffle
random.seed(1)


class BaseGenerator(object):
    """
    Generating data for data augmentation
    
    """
    def __init__(
            self,
            synonym_data_path: str = None,
            alpha: float = 0.2,
            num_aug: int = 4):
        """Instantiating Generator class

        Args:
            data_aug_method
        """
        super().__init__()
        self.alpha = alpha
        self.num_aug = num_aug
        self.alpha = alpha

        if synonym_data_path:
            self._synonym_df = get_tsv_file(synonym_data_path)

        self.num_new_per_technique = int(self.num_aug/4)+1

    def _get_df_row(self, name, word):
        row = self._synonym_df[self._synonym_df[name] == word]
        return row
    
    def _get_synonyms(self, word):
        """
        get a sysnonym word of my word
        """
        pass

    def _replace_synonym(self, words, n):
        """
        replace words with synonym words
        """
        pass

    def _add_synonym(self, words, new_words):
        """
        add synonym words into sentence
        """
        pass

    def _swap_word(self, words, new_words):
        pass

    def _delete_word(self, words, p):
        pass

    @staticmethod
    def tokenize_from_file(
        corpus_file,
        output_file,
        tokenizer_name,
        col_source: str = "org",
        col_name1: str = "pos",
        col_name2: str = "tagged"):

        tokenizer = get_tokenizer(tokenizer_name, pos=True)
        corpus = pd.read_csv(corpus_file, index_col=None, header=0, sep='\t', error_bad_lines=False)

        if "LABEL" in corpus.columns and "TEXT" in corpus.columns:
            corpus = corpus.rename(columns={"LABEL": "label", "TEXT": col_source})
        corpus[col_name1] = None
        corpus[col_name2] = None

        row_iterator = corpus.iterrows()

        print(">> Start Tokenizing... ")
        for i, row in tqdm(row_iterator, total=len(corpus)):
            sentence = tokenizer(row[col_source])
            s1 = [w[0] for w in sentence]
            s2 = [w[0] + "/" + w[1] for w in sentence]
            s1 = ' '.join(s1)
            s2 = ' '.join(s2)
            row[col_name1] = s1
            row[col_name2] = s2
            # if i > 100:
            #     break
        corpus.to_csv(output_file, mode='w', index=None, header=True, sep='\t')

    @property
    def corpus(self):
        return self._corpus_df

    @property
    def output(self):
        return self._output_file