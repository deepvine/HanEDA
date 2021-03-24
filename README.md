# HanEDA : Hangul Easy Data Augmentation
</h1>

Base paper: [EDA: Easy Data Augmentation techniques for boosting performance on text classification tasks.](https://arxiv.org/abs/1901.11196)

  
## Data Augmentation
기존 EDA 알고리즘(RD, RI, SR, RS)에 더해 한국어 특성에 맞게 알고리즘 수정
  - 조사 삭제
  - 조사 삭제 후 단어 스왑
  - 단순 문장 복사
  - 유사어 처리시 한국어 wordnet이 아닌 자체 유사어 사전 사용


## How to Install
```bash
python setup.py install
```

## Usage
```python
from haneda import run_algo

run_algo(
    corpus_file = corpus_file,
    output_file = output_file,
    synonym_file = synonym_file,
    mode = mode,
    max_value = max_value
)

```

## TO-DO list