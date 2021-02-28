from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

"""
Args:
    - name: pos tagger name
"""
def get_tokenizer(name: str, pos: bool=False):
    if name == "komoran":
        komoran = Komoran()
        if pos:
            tokenizer = komoran.pos
        else:
            tokenizer = komoran.morphs
    elif name == "okt":
        okt = Okt()
        if pos:
            tokenizer = okt.pos
        else:
            tokenizer = okt.morphs
    elif name == "mecab":
        mecab = Mecab()
        if pos:
            tokenizer = mecab.pos
        else:
            tokenizer = mecab.morphs
    elif name == "hannanum":
        hannanum = Hannanum()
        if pos:
            tokenizer = hannanum.pos
        else:
            tokenizer = hannanum.morphs
    elif name == "kkma":
        kkma = Kkma()
        if pos:
            tokenizer = kkma.pos
        else:
            tokenizer = kkma.morphs
    else:
        tokenizer = lambda x : x.split()
    return tokenizer