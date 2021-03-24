import glob
import argparse
from tqdm import tqdm
import pandas as pd


class Args:
    corpus_file: str = "/data/data/poly/ko2/ko-open-domain.tsv"
    output_file: str = "/data/data/poly/ko2/output.tsv"
    synonym_file: str = "data/mecab_synonym8.tsv"
    max_value: int = 10 # 생성 최대 값


class Args2:
    corpus_file: str = "data/tokenized2.tsv"
    output_file: str = "data/eda_output_test.tsv"
    synonym_file: str = "data/mecab_synonym8.tsv"
    max_value: int = 30 # 생성 최대 값
    logger_level: str = "DEBUG"

def main(args):
    from src import run_algo
    try:
        run_algo(
            corpus_file = args.corpus_file,
            output_file = args.output_file,
            synonym_file = args.synonym_file,
            mode = 2,
            max_value = args.max_value,
            logging_level = args.logger_level
        )
    except ValueError as exp:
        print ("Error", exp)


if __name__ == "__main__":
    args = Args2()
    main(args)
