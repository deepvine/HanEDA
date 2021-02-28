import re
import pandas as pd
from tqdm import tqdm
from utils import is_nan
from tokenizer import get_tokenizer

def _post_processing(tokens):
    results = []
    for token in tokens:
        # 숫자에 공백을 주어서 띄우기
        processed_token = [
            el for el in re.sub(r"(\d)", r" \1 ", token).split(" ") if len(el) > 0
        ]
        results.extend(processed_token)
    return results

def tokenize_from_tsv(
    tokenizer_name: str,
    input_path: str,
    output_path: str,
    y_index:int = 0,
    x_index:int = 1,
    y_header:str = "label",
    x_header:str = "text") -> None:
    """
    Tokenizing on input_path file and saving to output_path file

    Args:
        
    """

    tokenizer = get_tokenizer(tokenizer_name)
    df = pd.read_csv(input_path, header=0, sep="\t")
    total = len(df)
    print(">> Strart Tokenizing This File Like Below...")
    print(df.head(-10))

    with open(output_path, "w", encoding="utf-8") as f1:
        f1.writelines(y_header + "\t" + x_header + "\n")
        row_iterator = df.iterrows()

        for index, row in tqdm(row_iterator, total=total):
            sentence = row[x_index]
            label = row[y_index]

            if is_nan(sentence) or is_nan(label):
                continue
            replaced = label.replace(" ", "_")
            sentence = sentence.replace("\n", "").strip()

            tokens = tokenizer(sentence)
            tokenized_sent = " ".join(_post_processing(tokens))
            if is_nan(tokens) or tokens == "":
                continue

            f1.writelines(replaced + "\t" + tokenized_sent + "\n")
    f1.close()


tokenize_from_tsv(
    "mecab",
    "data/aihub_food.tsv",
    "data/tokenized.tsv"
)