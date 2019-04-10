import numpy as np
from typing import Sequence, Tuple, Any, Dict
from numba import njit
from joblib import Parallel, delayed

DocID = int
Doc = Sequence[DocID]




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
def _sim(a, b):
    return a.dot(b)

@njit
def _embed(embeddings, doc):
    v = np.zeros(embeddings.shape[0])
    for idx in doc:
        v += embeddings[:, idx]
    norm = np.linalg.norm(v)
    if norm == 0.:
        norm = 1.
    return v / norm

@njit
def _mean_embed(embeddings, docs):
    embeds = np.zeros((len(docs), embeddings.shape[0]))
    for i,n in enumerate(docs):
        embeds[i,:] = _embed(embeddings, n)
    mean = np_mean(embeds, 0)
    return mean

@njit
def _loss(embeddings, a, pos, negs):
    ea,epos = _embed(embeddings, a), _embed(embeddings, pos)
    neg_mean = _mean_embed(embeddings, negs)

    loss = 0.05 - _sim(ea,epos) + _sim(ea, neg_mean)
    loss = np.minimum(1e6, np.maximum(loss, 0.0))

    gradW = neg_mean - epos
    return loss, gradW

@njit
def _backward(step, grad_histories, embeddings, doc:Doc, grad):
    for idx in doc:
        grad_histories[idx] += grad.dot(grad)
        g = grad / (1e-6 + np.sqrt(grad_histories[idx]))
        embeddings[:, idx] -= step*grad

def _batch(step, grad_histories, embeddings, examples):
    loss = 0.
    gradients = []
    dim = embeddings.shape[1]
    for a,pos,negs in examples:
        # convert negs to sparse BOW
        l, gW = _loss(embeddings, a, pos, negs)
        loss += l
        gradients.append(gW)

    for i,e in enumerate(examples):
        a,pos,negs = e
        gW = gradients[i]
        _backward(step, grad_histories, embeddings, a, gW)

    return loss

def _epoch(step, batch_size, grad_histories, embeddings, examples):
    n = len(examples)

    batches = _make_batch(examples, batch_size)

    losses = Parallel(n_jobs=-1, require='sharedmem')(delayed(_batch)(step, grad_histories, embeddings, b)
                                                      for b in batches)
    loss = sum(losses)

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
