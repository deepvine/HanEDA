import pandas as pd

class DataDuplicator(object):
    """
    데이터를 늘리는 클래스
    """
    @staticmethod
    def duplicate(sentences: list, n: int):
        sl = len(sentences)
        choosed = [sentences[i % sl] for i in range(n)]
        return choosed

    @staticmethod
    def duplicate_token(sentence: str, n: int = 5):
        """토큰이 하나일 경우 단어를 n 값 만큼 늘리는 작업

        Args:
            n: 늘려야할 기준 값
        
        토큰이 하나인 경우: n 만큼 늘린다
            ex) 챗봇 -> 챗봇 챗봇 챗봇 챗봇 챗봇
        """
        hasPunc = False
        punc = ""
        s = sentence.split()
        if len(s) < 1:
            raise Exception()

        if s[-1] == ".":
            punc = s[-1]
            s = s[:-1]
            hasPunc = True

        if len(s) == 1:
            sentence = (s[0] + " ") * (n - 1) + s[0]
        elif len(s) == 2:
            sentence = (s[0] + " " + s[1] + " ") * int((n - 1)/2) + s[0] + " " + s[1]
        
        if hasPunc:
            sentence += punc
        return sentence