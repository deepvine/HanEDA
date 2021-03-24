import math
import numpy as np
import pandas as pd
import logging, traceback
from src.ko import KoGenerator
from .duplicator import DataDuplicator
from tqdm import tqdm
from .utils import *
from .tokenizer import get_tokenizer

# LABEL = "label"
# TEXT = "text"
# LABEL = "Answer"
# TEXT = "Question"
# CHATBOT = "chatbot"
# ORG = "origin"
# POS = "wpos"
# TAG = "tagged"

EDA = "eda_total"
SWP_DEL = "swap_delete"
SYN = "synonym_change"
DUP = "duplication"

global logger 

def init_logger(logging_level, name=None):
    logging.basicConfig(level=logging_level)
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)
    return logger

def save_text(
    corpus_file: str,
    output_file: str,
):
    """ save text columns from corpus file
    """
    corpus = pd.read_csv(corpus_file, header=None, sep="\t")

    output = open(output_file, "w+", encoding="utf-8")

    try:
        row_iterator = corpus.iterrows()
        for i, row in tqdm(row_iterator, total=len(corpus)):
            sentence = DataDuplicator.duplicate_token(row[1])
            output.writelines("{}\t{}\n".format(row[0], sentence))
    except Exception as e:
        logger.error(traceback.format_exc())


def get_numbers_to_generate(
    mode: int,
    need_to_gen: int,
    alpha: float,
    beta: float
):
    obj = dict()
    n_swdel, n_syn, n_dp = 0, 0, 0
    if mode == 0: # only copy
        n_dp = need_to_gen
    elif mode == 1: # swap, delete, synonym insert, synonym replace
        n_swdel = need_to_gen
    elif mode == 2: # swap, delete, synonym
        n_swdel = need_to_gen
    elif mode == 3: # synonym insert, synonym replace
        n_syn = need_to_gen
    else: # swap, delete, synonym insert, synonym replace
        n_eda = math.ceil(need_to_gen * alpha)
        n_swdel = math.ceil(n_eda * beta)
        n_syn = n_eda - n_swdel
        n_dp = need_to_gen - n_eda
    n_eda = n_swdel + n_syn
    # obj[EDA], obj[SWP_DEL], obj[SYN], obj[DUP] = \
    #     n_eda, n_swdel, n_syn, n_dp
    # # return obj
    return n_eda, n_swdel, n_syn, n_dp


def run_algo(
        corpus_file: str,
        output_file: str,
        synonym_file: str,
        mode: int = 4,
        max_value: int = 30,
        alpha: float = 0.8, # eda : dup
        beta: float = 0.7, # swap_del : synonym_change
        tokenizer: str = "mecab",
        **kwargs
    ):
    """EDA 알고리즘 실행
    EDA 알고리즘을 사용하여 corpus에 대하여 Data Augmentation 을 진행합니다
    EDA paper: https://arxiv.org/abs/1901.11196
    
    Args:
        corpus_file(str): DA가 진행될 소스 데이터 file path
            Data는 아래와 같이 구성되어 있어야 합니다
            LABEL, TEXT1, TEXT2, TEXT3
            TEXT1: swap_and_delete_word()에서 사용 됩니다
            TEXT3: synonym_insert_and_replace에서 사용 됩니다
        output_file(str): DA 결과가 저장될 타겟 file path
        synonym_file(str): 유사어 사전 파일(tsv)
        mode
            0: only run DUPLICATION
            1: only run DATA AUGMENTATION
            2: run both 0 and 1
        max_value: 얼마나 많은 utterance를 생성해야되는지 기준 값
    
    KoGenerator: 한국어 생성기(exobrain(a.k.a exo v1) or hancom(a.k.a exo v2))
    EngGenerator: 영어 생성기(nltk)

    max_value를 기준으로 하여 인텐트의 utterance 개수가 max_value보다 많으면 DA를 진행하지 않습니다.
        예를들어, max_value가 30인데, 현재 인텐트의 utterance가 32개 있으면 DA를 진행하지 않습니다.

    max_value보다 적으면 "필요한 개수"만큼 DA를 진행합니다. "필요한 개수"는 아래와 같이 계산됩니다.
        예를들어, max_value가 30인데, 현재 인텐트의 utterance가 27개 있으면 3개만 DA를 진행합니다.
    
    1) DUPLICATION MODE(MODE == 0)
        "필요한 개수" 만큼 데이터를 복제합니다.
        예를들어, 필요한 개수가 12개이면, 12개를 해당 intent에서 "순차적으로" 복제하여 30개를 채웁니다.
        "순차적"이라는 의미는 필요한 개수가 만족될 때까지 맨 위의 인텐트부터 차례대로 하나씩 복제합니다.
    2) DATA AUGMENTATION MODE (MODE == 1)
        "필요한 개수" 만큼 EDA를 실행합니다.
        상세한 알고리즘 실행은 KoGenerator.swap_and_delete_word()와
        KoGenerator.synonym_insert_and_replace()에서 실행됩니다.
    3) MIX MODE - ONLY SWAP AND DELETE  (MODE == 2)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 이므로
        14개를 복제하고 6개를 DA 합니다.
        다만, DA의 경우, KoGenerator.swap_and_delete_word()만 실행합니다.
    4) MIX MODE (MODE == 3)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 이므로
        14개를 복제하고 6개를 DA 합니다.
        다만, DA의 경우, KoGenerator.synonym_insert_and_replace()만 실행합니다.
    5) MIX MODE (MODE == 4)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 이므로
        14개를 복제하고 6개를 DA 합니다.
    """
    logger = init_logger(**kwargs)
    do_dup, do_da = False, False
    if mode < 5:
        if mode == 0:
            do_dup = True
        elif mode == 1:
            do_da = True
        else:
            do_dup = True
            do_da = True
    else:
        logger.error("Choose the generation mode between 0, 1 and 2")

    generator = KoGenerator(
        synonym_file = synonym_file,
        tokenizer = tokenizer
    )

    corpus = pd.read_csv(corpus_file, header=None, sep="\t")
    
    output = open(output_file, "w", encoding="utf-8")
    # output.writelines("{}\t{}\n".format(LABEL, TEXT))
    groups = corpus.groupby(corpus[0])
    intents = groups.groups
    group_size = groups.size()
    keys = list(groups.groups.keys())

    logger.info(">> Start Generating... ")
    logger.info("BEFORE DATA AUG...")
    logger.info("\nLABELS:\t{}\nTEXTS:\t{}\nMAX:\t{}\nMIN:\t{}\nAVG:\t{}\nSTD:\t{}\nMED:\t{}"\
        .format(len(keys), len(corpus), max(group_size), min(group_size), np.mean(group_size), np.std(group_size), np.median(group_size)))
    logger.info("end")

    try:
        row_iterator = corpus.iterrows()
        logger.info(">> Copy exist data... to output file(involved a work duplicating a single token) ")
        for i, row in tqdm(row_iterator, total=len(corpus)):
            sentence = DataDuplicator.duplicate_token(row[2])
            output.writelines("{}\t{}\n".format(row[0], sentence))
                

        for key in tqdm(keys, total=len(keys)):
            no_hit_synonym = 0
            intent_len = len(intents[key]) # 의도 utterance 개수
            if intent_len > max_value: # 이미 utterance가 max값 보다 많으면 skip
                continue
            need_to_gen = max_value - intent_len
            n_eda, n_swdel, n_syn, n_dp = get_numbers_to_generate(mode, need_to_gen, alpha, beta)
            total_no_hit_synonym = 0

            if do_da and n_swdel + n_syn > 0: # run DA
                if intent_len <= max_value:
                    augmentated = []
                    labels = []

                    # EDA 목표 값 안에서 의도 그룹 안에서 
                    # 각 문장마다 몇 개씩 DA를 해야하는지 계산
                    swdel_list = [0] * min(intent_len, n_swdel)
                    for i in range(n_swdel):
                        swdel_list[i % len(swdel_list)] += 1

                    syn_list = [0] * min(intent_len, n_syn)
                    for i in range(n_syn):
                        syn_list[i % len(syn_list)] += 1

                    # Data Augmentation
                    for i, intent in enumerate(list(intents[key])):
                        if i >= len(swdel_list) and i >= len(syn_list):
                            break

                        text, text2, text3 = None, None, None
                        row = corpus.loc[intent]
                        text = row[1]

                        if 2 in row.index:
                            text2 = row[2]
                        if 3 in row.index:
                            text2 = row[3]

                        n1 = swdel_list[i] if i < len(swdel_list) else 0
                        n2 = syn_list[i] if i < len(syn_list) else 0

                        if isHangul(text):
                            # Korean
                            if n1 > 0:
                                s = generator.swap_and_delete_word(text, n1)
                                if len(s) > 0:
                                    augmentated.extend(s)
                            if n2 > 0 and text3:
                                s, no_hit_synonym = generator.synonym_insert_and_replace(text3, n2)
                                total_no_hit_synonym += no_hit_synonym
                                if len(s) > 0:
                                    augmentated.extend(s)
                        else:
                            # English
                            pass
                    labels.extend([[row[0]]] * len(augmentated))

                    for sentence, label in zip(augmentated, labels):
                        sentence = DataDuplicator.duplicate_token(sentence)
                        output.writelines(
                            "{}\t{}\n".format(
                                label[0], sentence
                            )
                        )
                elif intent_len > n_eda:
                    n_dp = n_dp - (intent_len - n_eda)

            # Duplicate
            if do_dup:
                sentences = [corpus.loc[a][1] for a in intents[key]]
                duplicated = DataDuplicator.duplicate(sentences, n_dp + total_no_hit_synonym)
                for d in duplicated:
                    sentence = DataDuplicator.duplicate_token(d)
                    output.writelines(
                        "{}\t{}\n".format(
                            key, sentence
                        )
                    )

        output.close()

        output = pd.read_csv(output_file, header=None, sep="\t")
        groups = output.groupby(output[0])
        intents = groups.groups
        for intent in intents:
            if len(intents[intent]) < 30:
                logger.debug("{}, {}".format(intent, len(intents[intent])))
        group_size = groups.size()
        keys = list(groups.groups.keys())
        logger.info("")
        logger.info("AFTER DATA AUG...")
        logger.info("\nLABELS:\t{}\nTEXTS:\t{}\nMAX:\t{}\nMIN:\t{}\nAVG:\t{}\nSTD:\t{}\nMED:\t{}"\
            .format(len(keys), len(output), max(group_size), min(group_size), np.mean(group_size), np.std(group_size), np.median(group_size)))
        logger.info("end")

    except Exception as e:
        logger.error(traceback.format_exc())
