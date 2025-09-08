from __future__ import annotations        # allow forward type hints
from teenygrad.helpers import Dtype, dtypes, DEBUG   # dtype helpers + debug flag
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps  # op enums
import numpy as np                        # numpy for CPU array ops

# --- raw wrapper around a numpy array ---
class RawCPUBuffer:
    def __init__(self, x): self.x = x     # store numpy array
    def toCPU(self): return self.x        # return raw numpy array

# --- main tensor wrapper (CPU only here) ---
class LazyBuffer:
    device = "CPU"                        # hardcoded device

    def __init__(self, buf: np.ndarray): self._np = buf   # store numpy array in _np

    @property
    def base(self): return self           # base buffer (self on CPU)
    @property
    def dtype(self): return dtypes.from_np(self._np.dtype)  # map numpy dtype → teenygrad dtype
    @property
    def realized(self): return RawCPUBuffer(self._np)      # wrap data in RawCPUBuffer
    @property
    def shape(self): return self._np.shape # shape of numpy array
    def __repr__(self): return f"<LB {self.shape} {self.dtype}>" # debug print

    def schedule(self, seen=None): return []               # placeholder for lazy scheduling
    def is_unrealized_contiguous_const(self): return False # optimization stub
    def copy_to_device(self, device:str) -> LazyBuffer: return self # CPU only, so return self

    # --- creation helpers ---
    @staticmethod
    def fromCPU(x): return LazyBuffer(x)   # wrap existing numpy array

    @staticmethod
    def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
        if op == LoadOps.RAND:   # random tensor
            return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        elif op == LoadOps.CONST:  # constant-filled tensor
            return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY:  # uninitialized tensor
            return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else: raise NotImplementedError(op)

    def contiguous(x): return x           # no-op on CPU
    def const(self, x) -> LazyBuffer:     # fill like self with constant x
        return LazyBuffer(np.full_like(self._np, x))

    def cast(self, dtype:Dtype, bitcast:bool=False):       # change dtype
        return LazyBuffer(self._np.view(dtype.np) if bitcast else self._np.astype(dtype.np))

    # --- elementwise ops ---
    def e(self, op, *srcs:LazyBuffer):
        if DEBUG >=1: print(op, self, srcs)                # optional debug print
        if op == UnaryOps.NEG: ret = -self._np             # negation
        elif op == UnaryOps.EXP2: ret = np.exp2(self._np)  # exp2
        elif op == UnaryOps.LOG2: ret = np.log2(self._np)  # log2
        elif op == UnaryOps.SIN: ret = np.sin(self._np)    # sine
        elif op == UnaryOps.SQRT: ret = np.sqrt(self._np)  # sqrt
        elif op == BinaryOps.ADD: ret = self._np + srcs[0]._np  # add
        elif op == BinaryOps.SUB: ret = self._np - srcs[0]._np  # subtract
        elif op == BinaryOps.MUL: ret = self._np * srcs[0]._np  # multiply
        elif op == BinaryOps.DIV: ret = self._np / srcs[0]._np  # divide
        elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np) # max
        elif op == BinaryOps.CMPLT: ret = self._np < srcs[0]._np          # compare less-than
        elif op == TernaryOps.WHERE: ret = np.where(self._np, srcs[0]._np, srcs[1]._np) # ternary select
        else: raise NotImplementedError(op)

        # return result as new LazyBuffer, pick correct dtype
        return LazyBuffer(ret.astype(self.dtype.np if len(srcs) == 0 else max(self.dtype, *[x.dtype for x in srcs]).np, copy=False))
    
    # --- reduction ops ---
    def r(self, op, new_shape):
        if DEBUG >= 1: print(op, self, new_shape)          # optional debug
        assert len(self.shape) == len(new_shape), "reduce shapes must have same dimensions"
        axis = tuple(i for i,(a, b) in enumerate(zip(self.shape, new_shape)) if a != b) # find reduced dims
        if op == ReduceOps.SUM: return LazyBuffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True)) # sum reduce
        elif op == ReduceOps.MAX: return LazyBuffer(self._np.max(axis, keepdims=True))                     # max reduce
        else: raise NotImplementedError(op)

    # --- movement ops ---
    def reshape(self, arg): return LazyBuffer(self._np.reshape(arg))   # reshape
    def expand(self, arg): return LazyBuffer(np.broadcast_to(self._np, arg)) # broadcast
    def shrink(self, arg): return LazyBuffer(self._np[tuple(slice(p[0], p[1], None) for p in arg)]) # slice
    def permute(self, arg): return LazyBuffer(self._np.transpose(arg)) # transpose
    def pad(self, arg): return LazyBuffer(np.pad(self._np, arg))       # pad
    def stride(self, arg): return LazyBuffer(self._np[tuple(slice(None, None, i) for i in arg)]) # strided slice
