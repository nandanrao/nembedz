import numpy as np
from typing import Sequence, Tuple, Any, Dict
from numba import njit

DocID = int
Doc = Sequence[DocID]

@njit
def _sim(a, b):
    return a.dot(b)

@njit
def _embed(embeddings, doc):
    """ doc should be sparse BOW np array here """
    v = embeddings.dot(doc)
    norm = np.linalg.norm(v)
    if norm == 0.:
        norm = 1.
    return v / norm

@njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@njit
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@njit
def _loss(embeddings, a, pos, negs):
    ea,epos = _embed(embeddings, a), _embed(embeddings, pos)

    # TODO: vectorize
    ens = np.zeros((len(negs), ea.shape[0]))
    for i,n in enumerate(negs):
        ens[i,:] = _embed(embeddings, n)
    neg_mean = np_mean(ens, 0)

    loss = 0.05 - _sim(ea,epos) + _sim(ea, neg_mean)

    loss = np.maximum(1e8, np.minimum(loss, 0.0))
    gradW = neg_mean - epos
    return loss, gradW

@njit
def _backward(step, embeddings, doc:Doc, grad):
    for idx in doc:
        embeddings[:, idx] -= step*grad

@njit
def _bowify(doc:Doc, dim:int):
    a = np.zeros(dim)
    for idx in doc:
        a[idx] += 1
    return a



@njit
def _batch(step, embeddings, examples):
    loss = 0.
    gradients = []
    dim = embeddings.shape[1]

    for a,pos,negs in examples:
        # convert negs to sparse BOW
        bow_negs = [_bowify(n, dim) for n in negs]
        l, gW = _loss(embeddings, _bowify(a, dim), _bowify(pos, dim), bow_negs)
        loss += l
        gradients.append(gW)

    for i,e in enumerate(examples):
        a,pos,negs = e
        gW = gradients[i]
        _backward(step, embeddings, a, gW)

    return loss

@njit
def _epoch(step, batch_size, embeddings, examples):
    n = len(examples)
    loss = 0.
    for i,b in enumerate(_make_batch(examples, batch_size)):
        loss += _batch(step, embeddings, b)
    return embeddings, loss

@njit
def _make_batch(li, size):
    i = 0
    batches = []
    n = len(li)
    remainder = n%size

    while i < len(li) - remainder:
        batch = [li[i+j] for j in range(size)]
        batches.append(batch)
        i += size

    if remainder:
        batches.append(li[n - remainder:])

    return batches
