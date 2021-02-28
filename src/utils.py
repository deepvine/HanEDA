import json, os, re
from pathlib import Path
import pandas as pd

def is_nan(num):
    return num != num


def get_tsv_file(file_path: str):
    """
    Get DataFram object of Pandas library from file_path
    
    Args:
        file_path: a file path to need to get
    
    Return:
        a DataFrame instance
    """
    file_path_obj = Path(file_path)
    df = pd.read_csv(
        get_absolute_path(file_path_obj),
        header=0,
        sep='\t'
    )
    return df


def get_absolute_path(path):
    if isinstance(path, Path):
        path_obj = path
    else:
        if isinstance(path, str):
            path_obj = Path(path)
        else:
            raise RuntimeError("unusable path")
    return path_obj.absolute().as_posix()


def decimal_point(value):
    return value / 100


def isHangul(text):
    result = re.sub(re.compile('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', re.UNICODE), ' ', text)
    if result.strip():
        return True
    return False