import pandas as pd

class DataDuplicator(object):
    """
    """
    @staticmethod
    def duplicate(sentences: list, n: int):
        sl = len(sentences)
        choosed = [sentences[i % sl] for i in range(n)]
        return choosed

    @staticmethod
    def one_to_many(sentence: str, n: int = 5):
        """토큰이 하나일 경우 단어를 n 값 만큼 늘리는 작업

        Args:
            n: 늘려야할 기준 값
        
        1) 토큰이 하나인 경우: n 만큼 늘린다
            ex) 챗봇 -> 챗봇 챗봇 챗봇 챗봇 챗봇
        2) 토큰이 두개인 경우:
            의도데이터 포맷이 CHATBOT_NAME UTTERANCE라는 가정 하에
            token[0]을 제외한 token[1]을 n만큼 늘린다
            ex) 챗봇 안녕 -> 챗봇 안녕 안녕 안녕 안녕 안녕
        """
        s = sentence.split()
        if len(s) == 1:
            sentence = (s[0] + " ") * (n - 1) + s[0]
        elif len(s) == 2 and s[1] == ".":
            sentence = (s[0] + " ") * n + " " + s[1]
        return sentence