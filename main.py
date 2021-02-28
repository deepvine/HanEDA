import glob
import argparse
from tqdm import tqdm
import pandas as pd


class Args:
    corpus_file: str = "data/tokenized2.tsv"
    output_file: str = "data/eda_output_test.tsv"
    synonym_file: str = "data/mecab_synonym8.tsv"
    max_value: int = 30 # DA 생성 최대 값

def main(args):
    from src import KoGenerator, data_aug_for_textcnn
    try:
        data_aug_for_textcnn(
            args.corpus_file,
            args.output_file,
            args.synonym_file,
            max_value = args.max_value
        )
    except ValueError as exp:
        print ("Error", exp)


if __name__ == "__main__":
    args = Args()
    main(args)
