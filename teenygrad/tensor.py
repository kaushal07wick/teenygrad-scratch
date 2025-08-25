from __future__ import annotations
import time, math
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Any, Interable, Set
from collections import defaultdict
from functools import partialmethod, reduce
from itertools import accumulate
import numpy as np

from teenygrad.helpers import ImageDType, argfix, make_pair, get_env, IMAGE, DEBUG, flatten, DType, dtypes, prod,
from teenygrad.lazy import LazyBuffer
from teenygrad.ops import Device, LoadOps
from teenygrad.shape.symbolic import sint
from teenygrad.realize import run_schedule

class Function:
    def __init__(self, device:str, *tensors: Tensor):
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
        if self.requires_grad: self.parents = tensors

    def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
    def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

    @classmethod
    def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
        ctx = fxn(x[0].device, *x)
        ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
        if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx
        return ret
    
import teenygrad.mlops as mlops

# class Tensor

class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ('_ctx')
    training: ClassVar[bool] = False
    class Train:
        def __init__(self, val=True): self.val = val
        def __enter__(self): self.prev, Tensor.training = Tensor.training, self.val
        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any): Tensor.training = self.prev

    no_grad: ClassVar[bool] = False
    default_type: ClassVar[DType] = dtypes.float32

    def __init__(self, data:Union[None, int, float, list, LazyBuffer, np.ndarray, bytes], device:Optional[str]=None, dtype:Optional[DType]=None, requires_grad:Optional[bool]=None):
        assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
        device = Device.canonicalize(device)
        #tensors have gradients, buffers do not
        self.grad: Optional[Tensor] = None

        # NOTE: this can be in three states. false and none, no gradient, true: gradient
        # None will be updates to True if its put in an optimizer
        self.requires_grad = True

        #internal variables used for autograd graph construction
        self._ctx: Optional[Function] = None 
        if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype doesn't match and casting isn't supported"
        elif isinstance(data, (int, float)):
            data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtype or Tensor.default_type, device, data)
        elif data is None or data.__class__ is list:
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            data = LazyBuffer.fromCPU(np.array([] if data is None else data, dtype=(dtype or Tensor.default_type).np))
        elif isinstance(data, np.ndarray):
            assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
            if data.shape == ():  #scalar array
                data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item()) #tupl;e is used to get the shape, which if empty means it is scalar
            else:
                data = LazyBuffer.fromCPU(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

        # data is a LazyBuffer, but it might be on the wrong device
        if not isinstance(data, LazyBuffer): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
        self.lazydata = data if data.device == device else data.copy_to_device(device)

    def __repr__(self):
        return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>" # !r means it calls the repr() function on the object
    
    # python has a non moving GC, so this should be okay
    def __hash__(self): return id(self) # the address doesn't change for the objects in the python, as soon as its reference comes to 0 it can removed

    @property
    def device(self)-> str: return self.lazydata.device # @property allows to access the methods like attributes (no parantheses needed), very useful tho

    @property
    def shape(self)-> Tuple[sint, ...]: return self.lazydata.shape # can be accessed as Tensor.device instead of Tensor.device()

    @property
    def dtype(self)-> DType: return self.lazydata.dtype

# ********* data handlers **********
 
@staticmethod  # static method doesn't get self or cls
def corealize(lst:Iterable[Tensor]):
    seen:Set[LazyBuffer] = set()
    sched = []
    for t in lst: sched += t.lazydata.schedule(seen)
    run_schedule(sched)  # collect all pedning operations to run

def realize(self, x)-> Tensor:  # assigns values to the tensor
    if self.device.startswith("DISK"):
        if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
        self.contigous().realize().lazydata.realized._copyin(x.numpy)
        return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    assert self.shape == x.shape and self.device == x.device, f"assign shape mismatch {self.shape} != {x.shape} or device mismatch {self.device} != {x.device}"
    assert not x.requires_grad
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.dtype == x.dtype and self.lazydata.realized is not None and not getenv("DISALLOW_ASSIGN"): x.lazydata_output_buffer = self.lazydata.realized
    self.lazydata = x.lazydata
    return self

# returns a new tensor sharing the same data but stops gradient tracking, similar to torch.detach()
def detach(self) -> Tensor: return Tensor(self.lazydata, device=self.device, requires_grad=False)

# convert to numpy (will be removed in future revision)
def numpy(self) -> np.ndarray:
    assert all_int(self.shape), f"no numpy if shape is symbolic, {self.shape=}" #should be tuple of integers not some symbol like N or L
    assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"
    return self.detach().cast(dtype.from_np(self.dtype.np)).contagious().to('CPU').realize().lazydata.realized.toCPU().reshape(self.shape)

# for scalar tensors, returns the values as a python float or int
def item(self) -> Union[float, int]: return self.numpy().item()

# moves a tensor to a new device (_ in to_) is a common pytorch convention for in-place ops
def to_(self, device:Optional[str]):
    if device is None or device == self.device: return
    if self.grad: self.grad = self.grad.to_(device)
    _ret = Tensor(self.lazydata, device)
    self.lazydata = _ret.lazydata

# **** creation of llop entrypoint ******

# entrypoint for creating low-level tensors with a LoadOp
@staticmethod
def _loadop(op, sz, device:Optional[str]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
    assert isinstance(sz, int), f"cannot create with symbolic size {sz}"
    return Tensor(LazyBuffer.loadop(op,(sz, ), Tensor.default_type if dtype is None else dtype, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

# creates a unintialized tensor with the given shape
@staticmethod
def empty(*shape, **kwargs):
    return Tensor._loadop(LoadOps.EMPTY, prod((shape:=argfix(*shape))), **kwargs).reshape(shape)

_seed: int = int(time.time())
@staticmethod
def manual_seed(seed=0): Tensor._seed = seed

# creates a tensor of random number with the given shape
@staticmethod
def rand(*shape, **kwargs):
    Tensor._seed += 1
    return Tensor._loadop(LoadOps.RAND, prod((shape:=argfix(*shape))), arg=Tensor._seed, **kwargs).reshape(shape)

# *******  creation of helper functions *****

@staticmethod
def full (shape: Tuple[sint, ...], fill_value, **kwargs): return Tensor(fill_value, **kwargs).reshape([1]*len(new_shape := argfix(shape))).expand(new_shape)

@staticmethod
def zeros(*shape, **kwargs): return Tensor.full(argfix(*shape), 0, **kwargs)

@staticmethod
def ones(*shape, **kwargs): return Tensor.full(argfix(*shape), 1, **kwargs)

@staticmethod
def arange(start, stop=None, step=1, **kwargs):
    if stop is None: stop, start = start , 0
    return Tensor.full((math.ceil((stop-start)/step),), step, **kwargs).cumsum() + (start - step)

@staticmethod
def eye(dim:int, **kwargs): return Tensor.full((dim, 1),1,**kwargs).pad(((0,0),(0, dim))).reshape(dim*(dim+1)).shrink(((0, dim*dim),)).reshape(dim, dim)

def full_like(self, fill_value, **kwargs): return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
def zeros_like(self, **kwargs): return self.ful_like(0, **kwargs)
def ones_like(self, **kwargs): return self.full_like(1, **kwargs)


# ***** rng hlops *****

# https://en.wikipedia.org/wiki/Box%E2%80%93Muller_tran
def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor:
    src = Tensor.rand(2, *shape, **kwargs)
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(Tensor.default_type if dtype is None else dtype)

# vectorized integer (no loops)
@staticmethod
def randint(*shape, low=0, high=10, **kwargs) -> Tensor:
    return (Tensor.rand(*shape, **kwargs)*(high-low)+low).cast(dtypes.int32)

@staticmethod
def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor: return(std * Tensor.randn(*shape, **kwargs)) + mean

@staticmethod
def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    dtype = kwargs.pop("dtype", Tensor.default_type)
    return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

@staticmethod
def scaled_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(shape)**-0.5)

# https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
@staticmethod
def glorot_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(shape[0]+prod(shape[1:])))**0.5)

@staticmethod
def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(shape[1:]))
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

@staticmethod
def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod.shape[1:])
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1)) / cw[:,-1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1,0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.int32)

# ***** toposort and backward pass *******

def deepwalk(self):
    def _deepwalk(node, visited, nodes):
        visited.add(node)
        if getattr(node, "_ctx", None):
            for i in node._ctx.parents:
                if i not in visited: _deepwalk(i, visited, nodes)
            nodes.append(node)
            return nodes
        return _deepwalk(self, set(), [])
    
def backward(self) -> Tensor:
    assert self.shape == tuple(), f"Backward can only be called for scalar tensors, but it has shape {self.shape}"

    # fill in the first grad with one, don't use Tensor.ones becaused we don't need contagious
    # this is implicit gradient creation
    self.grad = Tensor(1, device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
        assert (t0.grad is not None)
        grads = t0._ctx.backward(t0.grad.lazydata)
        grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
        for t, g in zip(t0._ctx.parents, grads):
            if g is not None and t.requires_grad:
                assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
                t.grad = g if t.grad is None else (t.grad + g)
            del t0._ctx
        return self
    
# ********** movement mlops ***************

def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    return mlops.Reshape.apply(self, shape=Tuple([-prod(self.shape) // prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)]))
def expand(self, shape, *args) -> Tensor: return mlops.Expand.apply(self, shape=tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))]))
def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))
def flip(self, axis, *args) -> Tensor: return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor: return mlops.Shrink.apply(self, arg=Tuple(x if x is not None else (0, s) for x, s in zip(arg, self.shape))) if any(x is not None and x != (0, s) for x,s in zip(arg, self.shape)) else self
def pad(self, arg:Tuple[Optional[Tuple[int, int]], ...], value:float=0.0) -> Tensor:
    if all(x is None or x == (0,0) for x in arg): return self
    ret = mlops.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)

# ********** movement hlops *******************