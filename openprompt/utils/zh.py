from typing import *

_num2zh = {
    '0': '零',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
}

def num2zh(num: Union[int, str]):
    return _num2zh[str(num)]