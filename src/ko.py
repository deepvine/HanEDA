from .base import BaseGenerator
import pandas as pd
from .utils import *
import time
import re, math
from pathlib import Path
from .tokenizer import get_tokenizer

stop_words_ko = ['', 'ㄴ', './SF', '.', '에서', '은', 'ㄹ', '이', '하', '어', '을', '를'
			'는', '가', '에', '있', '어야', '데', '고', 'ㄹ게', 'ㄹ까', '들',
			'야', '었', '는데', 'ㄴ다', '세요', '수', '주', '어요', '요', '을까']

JOSA = ['JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'] # mecab 기준

class KoGenerator(BaseGenerator):
    """ 한국어 Data Augmentation Class

    한국어의 경우 DA는 크게 2가지로 진행됩니다.
    1) self.swap_and_delete_word : 토큰 단위로 단어의 위치를 서로 바꾸거나, 단어를 삭제합니다
    2) self.synonym_insert_and_replace: 유사어를 삽입하거나 교체합니다.


    DA mode에 따라 위의 2가지 알고리즘을 조합하여 사용합니다.
    1) DATA AUGMENTATION MODE (MODE == 1)
        "필요한 개수" 만큼 EDA를 실행합니다.
        상세한 알고리즘 실행은 KoGenerator.swap_and_delete_word()와
        KoGenerator.synonym_insert_and_replace()에서 실행됩니다.
    2) MIX MODE - ONLY SWAP AND DELETE  (MODE == 2)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 이므로
        14개를 복제하고 6개를 DA 합니다.
        다만, DA의 경우, KoGenerator.swap_and_delete_word()만 실행합니다.
    3) MIX MODE (MODE == 3)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 이므로
        14개를 복제하고 6개를 DA 합니다.
        다만, DA의 경우, KoGenerator.synonym_insert_and_replace()만 실행합니다.
    4) MIX MODE (MODE == 4)
        70:30 비율로 복제와 DA, 둘 다 실행합니다.
        예를들어, 20개의 utterance가 생성이 필요하면 비율에따라 복제:DA = 14:6 이므로
        14개를 복제하고 6개를 DA 합니다.
    
    """
    def __init__(
            self,
            synonym_file: str,
            tokenizer: str = "mecab",
            **kwargs):
        """Instantiating Generator class

        Args:
            - synonym_file(str): 유사어 사전 파일 경로
            - tokenizer(str): 토크나이저 네임
        """
        super(KoGenerator, self).__init__(
            synonym_file,
            **kwargs
        )
        self.tokenizer = get_tokenizer(tokenizer, pos=True)
    
    def _get_synonyms(self, word):
        synonyms = []
        row = self._get_df_row("word1", word)
        if not row.empty:
            syns = row.iloc[0]['synonym']
            for syn in syns.split():
                synonyms.append(syn)
        return synonyms

    def _replace_synonym(self, words: list, n: int) -> tuple:
        synonyms = []
        new_words = words.copy()
        # word_list = list(set([word for word in words if word not in stop_words_ko]))
        num_replaced = 0
        for word in words:
            synonyms = self._get_synonyms(word)
            if len(synonyms) >= 1:
                synonym = synonyms[0]
                new_words = [synonym if nw == word else nw for nw in new_words]
                num_replaced += 1
            if num_replaced >= n: # only replace up to n words
                break

        #this is stupid but we need it, trust me
        if num_replaced == 0:
            return ([], n - num_replaced)
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')
        new_words = [word.split("/")[0] for word in new_words]
        return (new_words, n - num_replaced)

    def _add_synonym(self, words: list, n: int) -> tuple:
        count = 0
        new_words = words.copy()
        for nw in new_words:
            if count >= n:
                break
            synonyms = self._get_synonyms(nw)
            if len(synonyms) > 0:
                new_words.insert(0, synonyms[0])
                count += 1
        if count == 0:
            return ([], n - count)
        new_words = [word.split("/")[0] for word in new_words]
        return (new_words, n - count)

    def _swap_word(self, words, n):
        if len(words) == 1 or\
            (len(words) == 2 and words[1] == "."):
            return [words]

        result = []
        count = 0
        idx1 = len(words)-1
        while words[idx1] in stop_words_ko:
            if count > 10:
                return words
            if idx1 < 0:
                return words
            idx1 -= 1
            count += 1
        for i in range(n):
            new_words = words.copy()
            if i + 1 == len(new_words):
                return result
            idx2 = i
            if words[idx2] in stop_words_ko:
                continue
            new_words[idx1], new_words[idx2] = words[idx2], words[idx1] 
            result.append(new_words)
        return result

    def _delete_word(self, words, n):
        len_words = len(words)

        # obviously, if there's only one word, don't delete it
        if len_words == 1 or\
            (len_words == 2 and words[1] == "."):
            return [words]

        result = []
        i = 0
        for _ in range(n):
            if i == len_words:
                break
            new_words = words.copy()
            idx = len_words - i - 1
            new_words.pop(idx)
            result.append(new_words)
            i += 1

        return result

    def _delete_josa(self, words: list):
        """
        Args:
            - words(list): 단어들은 반드시 POS정보를 같이 와야합니다. 예) 가방/NNG
        """
        if len(words) == 1:
            return words[0][0], words[0]

        result = []
        result2 = []
        for word, tag in words:
            if tag in JOSA:
                continue
            result.append(word)
            result2.append([word, tag])

        return ' '.join(result), result

    def swap_and_delete_word(self, sentence: str, n: int) -> list:
        """EDA 알고리즘 중 swap, delete 실행
        original paper: https://arxiv.org/abs/1901.11196
        
        Args:
            sentence: string sentence to be tagged
            n: the number of how many it needs to make augmentated data
        Return:
            DA 생성 결과 리스트
        
        1) 조사 제거
            - 모든 조사를 제거한 문장을 학습 데이터에 추가합니다
        2) SWAP
            - 순차적으로 선택한 두 토큰을 swap 처리 합니다
            - ex) 만나서 반갑습니다
                => 반갑습니다 만나서
        3) SWAP2
            - 조사제거된 문장에서 2)을 실행합니다
        4) REMOVE
            - 순차적으로 단어를 삭제하여 학습 데이터에 추가합니다
        """
        result = []
        n = n - 1
        need_to_swap = math.ceil(n/2/2)
        need_to_swap2 = math.floor(n/2/2)
        need_to_delete = math.floor(n/2)

        # 끝에 .이 있으면 제거
        if list(sentence)[-1] == ".":
            sentence = sentence[:-1]

        # Remove all Josa
        words = self.tokenizer(sentence)
        n_sentence, n_words = self._delete_josa(words)
        result.append(n_sentence)

        # Swap
        words = sentence.split()
        augmented_sentences = []
        swaped_sentences = self._swap_word(words, need_to_swap)
        for s in swaped_sentences:
            augmented_sentences.append(' '.join(s))

        # Swap 2
        swaped_sentences = self._swap_word(n_words, need_to_swap2)
        for s in swaped_sentences:
            augmented_sentences.append(' '.join(s))

        # Remove a word
        deleted_sentences = self._delete_word(words, need_to_delete)
        for s in deleted_sentences:
            augmented_sentences.append(' '.join(s))

        for s in augmented_sentences:
            words = self.tokenizer(s)
            words = [word[0] for word in words]
            tagged = ' '.join(words)
            result.append(tagged)

        return result

    def synonym_insert_and_replace(self, sentence: str, n: int) -> (list, int):
        """EDA 알고리즘 중 유의어 추가, 교체 실행
        original paper: https://arxiv.org/abs/1901.11196
        
        Args:
            sentence: 입력문장(형태소 정보 포함되어 있어야됨)
            n: augmentation할 갯수
        Return:
            DA 생성 결과 리스트
        
        1) 유의어 추가
            - 문장에 유의어를 추가 합니다.
        2) 유의어 교체
            - 단어를 유의어로 변경합니다.
        """
        result = []
        no_hit = 0
        need_to_insert = math.ceil(n/2)
        need_to_replace = math.floor(n/2)

        # 끝에 .이 있으면 제거
        if list(sentence)[-1] == ".":
            sentence = sentence[:-1]

        # Insert
        if need_to_insert > 0:
            words = sentence.split()
            augmented_sentences = []
            added_sentence, no_hit_count = self._add_synonym(words, need_to_insert)
            if len(added_sentence) > 0:
                augmented_sentences.append(' '.join(added_sentence))
            no_hit += no_hit_count
            

        # Replace
        if need_to_replace > 0:
            changed_sentence, no_hit_count = self._replace_synonym(words, need_to_replace)
            if len(changed_sentence) > 0:
                augmented_sentences.append(' '.join(changed_sentence))
            no_hit += no_hit_count

        return augmented_sentences, no_hit