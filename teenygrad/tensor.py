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


def __getitem__(self, val) -> Tensor: # val : Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice]]]
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz: return e if e != -1 else dim_sz-1
        raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")
    
    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i, v in enumerate(orig_slices): count[type(v)].append(i)

    if (num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])) > len(self.shape): raise IndexError(f"too many indeices for tensor of dimension {len(self.shape)}")
    if len(ellipsis_found := count[type(ellipsis)]) > 1: raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [v if isinstance(v, slice) else slice(y_ := normalize_int(v, i, dim_sz), y_+1) if isinstance(v, int) else slice(None) for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))]

    start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((),(),())
    new_slice = tuple(((0,0) if e < s else (s, e)) if st > 0 else ((0,0) if e > s else (e+1, s+1)) for s, e, st in zip(start, stop, strides))
    sliced_tensor = self.shrink(new_slice).flip(axis=[i for i,s in enumerate(strides) if s < 0])
    new_shape = sliced_tensor.shape
    if any(abs(s) != 1 for s in strides):
        strides = tuple(abs(s) for s in strides)
        #pad : add pad at the end: [dim_sz] -> [dim_sz_padded]
        padded_tensor = sliced_tensor.pad(tuple(0, s-(dim_sz % s) if dim_sz != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape))
        #Reshape : [dim_sz_padded] -> [dim_sz_padded //s, s]
        reshaped_tensor = padded_tensor.reshape(flatten([sh  // s, s] for sh, s in zip(padded_tensor.shape, strides)))
        new_shape = reshaped_tensor.shape[::2]
        # shrink : do [:, 0]
        sliced_tensor = reshaped_tensor.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

    final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
    for i,s in enumerate(orig_slices):
        if s is None: final_shape.append(1)
        else:
            dim_shape = next(it_shape)
            if isinstance(s, int):
                dim_collapsed += 1
            else:
                assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
                final_shape.append(dim_shape)
                if isinstance(s, Tensor):
                    tensors.append(s)
                    dim.append(s)
    ret = sliced_tensor.reshape(tuple(final_shape))


    if tensors:  # fancy/tensor indexing
        # normalize idx

        idx = [t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t for d, t in zip(dim, tensors)]
        max_dim = max(i.ndim for i in idx)
        # compute sum_dim, arange, and idx
        sum_dim = [d if n==0 else d+max_dim-n for n, d in enumerate(dim)]
        arange = [Tensor.arange(ret.shape[d], dtype=dtypes.int32, requires_grad=False, device=self.device).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n, (sd, d) in enumerate(zip(sum_dim, dim))]
        first_idx = [idx[0].reshape(*[1]*dim[0], *[1]*( + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - dim[0] - 1))]
        rest_idx = [i.reshape(*[1]*dim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - dim[0] - n)) for n, i in enumerate(idx[1:], 1)]
        idx = first_idx + rest_idx
        ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])
        #iteratively fancy index
        for a, i, sd in zip(arange, idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
        #special permute case
        if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1]+1)):
            ret_dims = list(range(ret.ndim))
            ret = ret.permute(ret_dims[dim[0]:dim[0]+max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0]+max_dim:])
    return ret
    
def __setitem__(self, s, v): return self.__getitem__(s).assign(v)

def slice(self, arg:Sequence[Optional[tuple[int, sint]]], value:float=0) -> Tensor:
    arg_ = tuple([a if a is not None else (0, s) for s, a in zip(self.shape, arg)])
    padding = tuple([(max(0, -p[0]), max(0, p[1]-self.shape[i])) for i, p in enumerate(arg_)])
    return self.pad(padding, value=value).shrink(tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)]))

def gather(self: Tensor, idx: Tensor, dim: int) -> Tensor:
    assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
    assert all(s >= i for s,i in zip(self.shape, idx.shape)), "all dim of idx.shape must be smaller than self.shape"
    if dim < 0: dim += self.ndim
    idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)
    permarg = list(range(self.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((idx == Tensor.arange(self.shape[dim], dtype=dtypes.int32, requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(tuple([*[(0,sh) for sh in idx.shape[1:-1]], (0,self.shape[dim])])).unsqueeze(0)).sum(-1).transpose(ax1=0, ax2=dim)

def cat(self, *args, dim=0) -> Tensor:
    dim = (dim + len(self.shape)) if dim < 0 else dim
    assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
    catargs = [self, *args]
    assert all(t.shape for t in catargs), "zero dimensional tensor cannot be concatenated"
    shapes = [s.shape[dim] for s in catargs]
    shape_cumsum = [0, *accumulate(shapes)]
    slc = [[(0, 0) for _ in self.shape] for _ in catargs]
    for shp, k, s in zip(shapes, shape_cumsum[:-1], slc): s[dim] = (k, shape_cumsum[-1] - k - shp)
    return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg, s in zip(catargs, slc)])

@staticmethod
def stack(tensors, dim=0) -> Tensor:
    first = tensors[0].unsqueeze(dim)
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
    return first.cat(*unsqueezed_tensors, dim=dim)

def repeat(self, repeats) -> Tensor:
    base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r**s for r, s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

def chunk(self, num:int, dim:int=0) -> List[Tensor]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim, step = dim + self.ndim if dim < 0 else dim, math.ceil(self.shape[dim]/num)
    slice_params = [[slice(None)]*dim + [slice(k, k + step)] for k in range(0, self.shape[dim], step)]
    return [self[tuple(sl)] for sl in slice_params]

def squeeze(self, num:int, dim:int=0) -> List[Tensor]:
    if dim is None: return self if 1 not in self.shape else self.reshape(*[size for size in self.shape if size != 1])
    if dim < 0 and self.ndim == 0: return self
    if not -self.ndim <= dim < self.ndim: raise IndexError(f"Dimension out of range (expected to be in range of [{-self.ndim if self.ndim > 0 else self.ndim-1}, {self.ndim-1 if self.ndim > 0 else self.ndim}], but got {dim})")
    if dim < 0: dim += self.ndim
    return self if self.shape[dim] != 1 else self.reshape(*[size for idx, size in enumerate(self.shape) if idx != dim])

def unsqueeze(self, dim) -> Tensor:
    if dim < 0: dim = len(self.shape) + dim + 1
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

def pad2d(self, padding:Union[List[int], Tuple[int, ...]], value:float=0) -> Tensor:
    slc = [(-p0, s+p1) for p0, o1, s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
    return self.slice([(0, s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)
