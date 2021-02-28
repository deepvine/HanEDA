import math
import numpy as np
import pandas as pd
from src.ko import KoGenerator
# from src.en import EngGenerator
from .duplicator import DataDuplicator
from tqdm import tqdm
from .utils import *
from .tokenizer import get_tokenizer

LABEL = "label"
TEXT = "text"
CHATBOT = "chatbot"
ORG = "org"
POS = "wpos"
TAG = "tagged"

EDA = "eda_total"
SWP_DEL = "swap_delete"
SYN = "synonym_change"
DUP = "duplication"


def save_text(
    corpus_file: str,
    output_file: str,
):
    """
    save text columns from corpus file
    """
    corpus = pd.read_csv(corpus_file, header=0, sep="\t")

    output = open(output_file, "w+", encoding="utf-8")
    output.writelines("{}\t{}\n".format(LABEL, TEXT))

    try:
        row_iterator = corpus.iterrows()
        for i, row in tqdm(row_iterator, total=len(corpus)):
            sentence = DataDuplicator.one_to_many(row[TEXT])
            if CHATBOT in corpus.columns:
                output.writelines("{}\t{}\n".format(row[LABEL], row[CHATBOT] + " " + sentence))
            else:
                output.writelines("{}\t{}\n".format(row[LABEL], sentence))
    except ValueError as exp:
        print("Error", exp)


def cal(
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
    obj[EDA], obj[SWP_DEL], obj[SYN], obj[DUP] = \
        n_eda, n_swdel, n_syn, n_dp
    return obj


def data_aug_for_textcnn(
        corpus_file: str,
        output_file: str,
        synonym_file: str,
        mode: int = 4,
        max_value: int = 30,
        alpha: float = 0.8, # eda : dup
        beta: float = 0.7 # swap_del : synonym_change
    ):
    """corpus에 대하여 Data Augmentation 을 진행합니다
    EDA 알고리즘을 사용하여 DA를 진행합니다
    EDA paper: https://arxiv.org/abs/1901.11196
    
    Args:
        corpus_file: DA가 진행될 소스 데이터 file path
        output_file: DA 결과가 저장될 타겟 file path
        synonym_file: 유사의 사전 파일(tsv)
        mode
            0: only run DUPLICATION
            1: only run DATA AUGMENTATION
            2: run both 0 and 1
        max_value: 얼마나 많은 utterance를 생성해야되는지 기준 값
    
    KoGenerator: 한국어 생성기(exobrain(a.k.a exo v1) or hancom(a.k.a exo v2))
    EngGenerator: 영어 생성기(nltk)

    max_value를 기준으로 하여 인텐트의 utterance 개수가 max_value보다 많으면 DA를 진행하지 않습니다.
    max_value보다 적으면 "필요한 개수"만큼 DA를 진행합니다. "필요한 개수"는 아래와 같이 계산됩니다.
    
    1) DUPLICATION MODE(MODE == 0)
        max_value - 인텐트 utterance 개수 = 필요한 개수 만큼 데이터를 복제합니다.
        ex) len(intent_A) = 18, max_value = 30 이면
            max_value - len(intent_A) = 12 이므로, 12개를 intent_A에서 순차적으로 복제하여 30개를 채웁니다.
    2) DATA AUGMENTATION MODE (MODE == 1)
        max_value - 인텐트 utterance 개수 = 필요한 개수 만큼 EDA를 실행합니다.
        (EDA 알고리즘 실행 절차는 KoGenerator.swap_and_delete_word() 을 참고바랍니다.)
    3) 혼합 MODE (MODE == 2)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 개수로 생성됩니다.
    4) 혼합 MODE (MODE == 3)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 개수로 생성됩니다.
    5) 혼합 MODE (MODE == 4)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 개수로 생성됩니다.
    """
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
        raise RuntimeError("Choose the generation mode between 0, 1 and 2")

    generator = KoGenerator(
        synonym_file = synonym_file,
        tokenizer = "mecab",
        alpha = 0.2
    )

    # eg = EngGenerator(
    #     tokenizer = "hancom", # deeplearning4j
    #     alpha = 0.2
    # )

    corpus = pd.read_csv(corpus_file, header=0, sep="\t")
    
    output = open(output_file, "w", encoding="utf-8")
    output.writelines("{}\t{}\n".format(LABEL, TEXT))
    groups = corpus.groupby(corpus[LABEL])
    intents = groups.groups
    group_size = groups.size()
    keys = list(groups.groups.keys())

    print(">> Start Generating... ")
    print("## BEFORE DATA AUG ###########")
    print("LABELS:\t{}\nTEXTS:\t{}\nMAX:\t{}\nMIN:\t{}\nAVG:\t{}\nSTD:\t{}\nMED:\t{}"\
        .format(len(keys), len(corpus), max(group_size), min(group_size), np.mean(group_size), np.std(group_size), np.median(group_size)))
    print("########################")
    print()

    try:
        row_iterator = corpus.iterrows()
        for i, row in tqdm(row_iterator, total=len(corpus)):
            sentence = DataDuplicator.one_to_many(row[TEXT])
            if CHATBOT in corpus.columns:
                output.writelines("{}\t{}\n".format(row[LABEL], row[CHATBOT] + " " + sentence))
            else:
                output.writelines("{}\t{}\n".format(row[LABEL], sentence))

        for key in tqdm(keys, total=len(keys)):
            no_hit_synonym = 0
            if do_da: # run DA
                intent_len = len(intents[key]) # 의도 utterance 개수
                if intent_len > max_value: # 이미 utterance가 max값 보다 많으면 skip
                    continue
                elif intent_len <= max_value:
                    augmentated = []
                    label_chatbot = []
                    need_to_gen = max_value - intent_len
                    count_object = cal(mode, need_to_gen, alpha, beta)

                    # EDA 목표 값 안에서 각 문장마다 몇 개씩 DA를 해야하는지 계산
                    # TODO
                    gen_list = [0] * min(intent_len, need_to_gen)
                    for i in range(need_to_gen):
                        gen_list[i % len(gen_list)] += 1

                    # Data Augmentation
                    for i, intent in enumerate(list(intents[key])):
                        # if i >= len(gen_list):
                        #     break

                        row = corpus.loc[intent]
                        # origin, text, tagged = row[ORG], row[POS], row[TAG]
                        origin = row[TEXT]
                        text_wpos = row[POS]

                        if isHangul(origin):
                            # Korean
                            if count_object[SWP_DEL] > 0:
                                s = generator.swap_and_delete_word(origin, count_object[SWP_DEL])
                            if count_object[SYN] > 0:
                                s, no_hit_synonym = generator.synonym_insert_and_replace(text_wpos, count_object[SYN])
                            if len(s) > 0:
                                augmentated.extend(s)
                        else:
                            # English
                            # s = eg.swap_and_delete_word(origin, n)
                            # augmentated.extend(s)
                            pass
                        if CHATBOT in corpus.columns:
                            lc = [[row[LABEL], row[CHATBOT]]] * len(s)
                        else:
                            lc = [[row[LABEL]]] * len(s)
                        label_chatbot.extend(lc)

                    for sentence, l_c in zip(augmentated, label_chatbot):
                        sentence = DataDuplicator.one_to_many(sentence)
                        if CHATBOT in corpus.columns:
                            output.writelines(
                                "{}\t{}\n".format(
                                    l_c[0], l_c[1] + " " + sentence
                                )
                            )
                        else:
                            output.writelines(
                                "{}\t{}\n".format(
                                    l_c[0], sentence
                                )
                            )
                elif intent_len > count_object[EDA]:
                    n_dp = n_dp - (intent_len - count_object[EDA])

            # Duplicate
            if do_dup:
                sentences = [corpus.loc[a][TEXT] for a in intents[key]]

                if CHATBOT in corpus.columns:
                    chatbot_name = corpus.loc[intents[key][0]][CHATBOT]
                    s = chatbot_name + " "
                else:
                    s = ""
                duplicated = DataDuplicator.duplicate(sentences, count_object[DUP] + no_hit_synonym)
                for d in duplicated:
                    sentence = DataDuplicator.one_to_many(d)
                    output.writelines(
                        "{}\t{}\n".format(
                            key, s + sentence
                        )
                    )

        output.close()

        output = pd.read_csv(output_file, header=0, sep="\t")
        groups = output.groupby(output[LABEL])
        intents = groups.groups
        for intent in intents:
            if len(intents[intent]) < 30:
                print('dd')
        group_size = groups.size()
        keys = list(groups.groups.keys())
        print()
        print("## AFTER DATA AUG ###########")
        print("LABELS:\t{}\nTEXTS:\t{}\nMAX:\t{}\nMIN:\t{}\nAVG:\t{}\nSTD:\t{}\nMED:\t{}"\
            .format(len(keys), len(output), max(group_size), min(group_size), np.mean(group_size), np.std(group_size), np.median(group_size)))
        print("########################")
        print()

    except ValueError as exp:
        print("Error", exp)
