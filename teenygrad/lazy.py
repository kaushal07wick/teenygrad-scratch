from __future__ import annotations
from teenygrad.helpers import Dtype, dtypes, DEBUG
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np

class RawCPUBuffer:
    def __init__(self, x): self.x = x
    def toCPU(self): return self.x

class LazyBuffer:
    device = "CPU"

    def __init__(self, buf: np.ndarray): self._np = buf

    @property
    def base(self): return self
    @property
    def dtype(self): return dtypes.from_np(self._np.dtype)
    @property
    def realized(self): return RawCPUBuffer(self._np)
    @property
    def shape(self): return self._np.shape
    def __repr__(self): return f"<LB {self.shape} {self.dtype}>"

    def schedule(self, seen=None): return []
    def is_unrealized_contiguous_const(self): return False
    def copy_to_device(self, device:str) -> LazyBuffer: return self

    @staticmethod
    def fromCPU(x): return LazyBuffer(x)

    @staticmethod
    def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
        if op == LoadOps.RAND: return LazyBuffer