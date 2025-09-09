from typing import Union, Tuple, Iterator, Optional, Final, Any
import os, functools, platform
import numpy as np
from math import prod
from dataclasses import dataclass

OSX = platform.system() == "Darwin"
def dedup(x): return list(dict.fromkeys(x)) # retains list order
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x, )*cnt if isinstance(x, int) else x
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))
def all_int(t: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in t)
def round_up(num, amt:int): return (num+amt-1)//amt * amt

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
CI = os.getenv("CI", "") != ""

@dataclass
class Dtype:
    priority: int
    itemsize: int
    name: str
    np: Optional[type]
    sz: int = 1
    def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
    @staticmethod   # static methods on top or bool in the type info will refer to dtypes.bool
    def is_int(x: Dtype) -> bool: return x in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
    @staticmethod
    def is_float(x: Dtype) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64)
    @staticmethod
    def is_unsigned(x: Dtype) -> bool: return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
    @staticmethod
    def from_np(x) -> Dtype: return DTYPES_DICT[np.dtype(x).name]
    bool: Final[Dtype] = Dtype(0, 1, "bool", np.bool_)
    float16: Final[Dtype] = Dtype(9, 2, "half", np.float16)
    half = float16
    float32: Final[Dtype] = Dtype(10,4, "float", np.float32)
    float = float32
    float64: Final[Dtype] = Dtype(11, 8, "double", np.float64)
    double = float64
    int8: Final[Dtype] = Dtype(1, 1, "char", np.int8)
    int16: Final[Dtype] = Dtype(3, 2, "short", np.int16)
    int32: Final[Dtype] = Dtype(5, 4, "int", np.int32)
    int64: Final[Dtype] = Dtype(7, 8, "long", np.int64)
    uint8: Final[Dtype] = Dtype(2, 1, "unsigned char", np.uint8)
    uint16: Final[Dtype] = Dtype(4, 2,  "unsigned short", np.uint16)
    uint32: Final[Dtype] = Dtype(6, 8, "unsigned long", np.uint64)
    uint64: Final[Dtype] = Dtype(8, 10, "unsigned long long", np.uint64)

    # NOTE: bfloat16 isn't supported in numpy
    bfloat16: Final[Dtype] = Dtype(9, 2, "__bf16", None)

DTYPES_DICT = {k: c for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}

PtrDType, ImageDType, IMAGE = None, None, 0  # junk to remove