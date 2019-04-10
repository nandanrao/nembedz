from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any, Dict
import csv
from numba import njit
import numpy as np
from gensim.corpora import Dictionary
from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords, strip_short
import re
from functools import reduce
from multiprocessing import Pool
from random import randrange, sample
from joblib import Parallel, delayed
from dataclasses import dataclass
from time import perf_counter

import random
random.seed(123)
np.random.seed(123)

from inner_python import _epoch

DocID = int
Doc = Sequence[DocID]

@dataclass
class Nembedz(ABC):
    dictionary:Dictionary
    dims:int = 100
    num_negs:int = 50
    batch_size:int = 5
    step_size:float = 0.01
    embeddings:np.ndarray = None

    def __post_init__(self):
        vocab_size = len(self.dictionary.keys())
        if not self.embeddings:
            self.embeddings = np.random.normal(0, .001, (self.dims, vocab_size))
        self.grad_histories = np.zeros(vocab_size)


    def _parse(self, doc:Sequence[str]) -> Sequence[int]:
        tokens = self.dictionary.doc2idx(doc)
        return [t for t in tokens if t != -1]

    def load_docs(self, docs:Sequence[str]):
        string_it = ((i, self.preprocess(s)) for i,s
                in enumerate(docs))
        with Pool() as pool:
            stringit = pool.map(self.preprocess, docs)
            docit = pool.map(self._parse, stringit)

        Corpus = Dict[int, Sequence[int]]
        self.corpus:Corpus = dict(((i,d) for i,d
                                   in enumerate(docit) if len(d)))

        self.doc_ids = np.array(list(self.corpus.keys()))

    def _epoch(self):
        ids = np.array(list(self.corpus.keys())).copy()
        np.random.shuffle(ids)

        print("Generating Samples")
        start_time = perf_counter()

        with Pool() as pool:
            positives = pool.map(self.positive, ids)
            negatives = pool.map(self.negatives, ids)

        examples = [(*p, n) for p,n in zip(positives, negatives)]
        examples = [(np.array(a), np.array(p), tuple([np.array(x) for x in n]))
                    for a,p,n in examples]

        print(f'Samples generated in {perf_counter() - start_time} seconds')
        print("Beginning Training")
        start_time = perf_counter()

        loss = _epoch(self.step_size,
                      self.batch_size,
                      self.grad_histories,
                      self.embeddings,
                      examples)


        print(f'Training completed in {perf_counter() - start_time} seconds')
        return loss

    @abstractmethod
    def preprocess(self, s:str) -> Sequence[str]:
        pass

    @abstractmethod
    def positive(self, i:int) -> Tuple[Doc, Doc]:

        # For within doc, pick whatever you want
        # within doc

        # For skipgram
        # just get window within doc, easy
        # access to frequencies for subsampling?

        # For user/sequence
        # from tags get "x"
        # get find index of doc
        # get next or prev index
        # return both

        # For tagspace
        # just get a random doc with the same tag(s?)

        pass

    @abstractmethod
    def negatives(self, i:int) -> Sequence[Doc]:
        # for tagspace
        # get a random doc with no shared tags

        # for user/sequence
        # get a random doc from another user

        # for others
        # get a random doc

        pass


patterns = {
    'urls': re.compile(r'https?:\/\/[^\s]+'),
    'RT' : re.compile(r'RT\s?@\w+\s?:?'),
    'mentions' : re.compile(r'@\w+'),
    'hashtags' : re.compile(r'#\w+')
}

def clean_tweet(s):
    for k,pat in patterns.items():
        s = re.sub(pat, '', s)
    s = ' '.join(s.split()).strip().lower()
    return s

class SimpleEmbedder(Nembedz):
    def preprocess(self, s):
        pipeline = [clean_tweet, remove_stopwords, strip_short, tokenize]
        l = list(reduce(lambda r,f: f(r), pipeline, s))
        return l

    def positive(self, i:int) -> Tuple[Doc, Doc]:
        doc = self.corpus[i]

        n = len(doc)

        # if only one word, embed it with itself...
        # basically does nothing except pushes it
        # away from everything else...
        if n == 1:
            return doc, doc

        i = randrange(1, n)
        return doc[:i], doc[i:]

    def negatives(self, i:int) -> Sequence[Doc]:
        s = np.random.choice(self.doc_ids, self.num_negs, replace=False)
        return [self.corpus[x] for x in s
                if x is not i]

def preprocess(s):
    pipeline = [clean_tweet, remove_stopwords, strip_short, tokenize]
    l = list(reduce(lambda r,f: f(r), pipeline, s))
    return l

def main(path, epochs):
    with open(path) as f:
        ds = csv.DictReader(f)
        tweets = (d['tweet_text'] for d in ds)
        tweets = list(tweets)

    dct = Dictionary()
    with Pool() as pool:
        dct.add_documents(pool.map(preprocess, tweets))
    dct.filter_extremes(no_below=10, no_above=0.5)

    model = SimpleEmbedder(dct, dims = 100)
    model.load_docs(tweets[-100000:])

    for i in range(int(epochs)):
        loss = model._epoch()
        print(f'Epoch {i} loss: {loss[1]}')
        np.save('embeddings', model.embeddings)

from clize import run

from numba import njit, prange
from math import floor


if __name__ == '__main__':
    run(main)
