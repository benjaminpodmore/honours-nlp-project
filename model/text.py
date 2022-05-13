import attr
import random
from cached_property import cached_property
from copy import deepcopy as c
from utils import compute_idx_spans, flatten
from boltons.iterutils import pairwise

@attr.s(frozen=True, repr=False)
class Span:
    i1 = attr.ib()
    i2 = attr.ib()
    id = attr.ib()
    speaker = attr.ib()
    si = attr.ib(default=None)
    yi = attr.ib(default=None)
    yi_idx = attr.ib(default=None)

    def __len__(self):
        return self.i2 - self.i1 + 1

    def __repr__(self):
        return f"Span representing {self.__len__()} tokens"


class Corpus:
    def __init__(self, documents):
        self.docs = documents
        self.vocab, self.char_vocab = self.get_vocab()

    def __getitem__(self, idx):
        return self.docs[idx]

    def __repr__(self):
        return "Corpus containing {len(self.docs)} documents"

    def get_vocab(self):
        vocab, char_vocab = set(), set()

        for document in self.docs:
            vocab.update(document.tokens)
            char_vocab.update([char for word in document.tokens for char in word])

        return vocab, char_vocab


class Document:
    def __init__(self, tokens, corefs, speakers, filename, raw_text=None, sentences=None):
        self.tokens = tokens
        self.corefs = corefs
        self.filename = filename
        self.raw_text = raw_text

        if sentences is None:
            sent_idx = [idx + 1
                        for idx, token in enumerate(self.tokens)
                        if token in ['.', '?', '!']]

            # Regroup (returns list of lists)
            sentences = [{'acc_length': i1, 'sentence': self.tokens[i1:i2]} for i1, i2 in pairwise([0] + sent_idx)]

        if speakers is None:
            speakers = ["-" for t in tokens]
        self.speakers = speakers

        self.sentences = sentences
        self.tags = None

    def __getitem__(self, idx):
        return self.tokens[idx], self.corefs[idx], self.speakers[idx]

    def __repr__(self) -> str:
        return f"Document containing {len(self.tokens)} tokens"

    def __len__(self):
        return len(self.tokens)

    @cached_property
    def sents(self):
        return [s['sentence'] for s in self.sentences]

    def spans(self):
        return [Span(i1=i[0], i2=i[-1], id=idx,
                     speaker=self.speaker(i))
                for idx, i in enumerate(compute_idx_spans(self.sents))]

    def truncate(self, MAX=15):
        """ Randomly truncate the document to up to MAX sentences """
        if len(self.sents) > MAX:
            i = random.sample(range(MAX, len(self.sents)), 1)[0]
            tokens = flatten(self.sents[i-MAX:i])
            sentences = self.sentences[i-MAX:i]
            speakers = self.speakers[
                       sentences[0]['acc_length']:sentences[-1]['acc_length'] + len(sentences[-1]['sentence'])]

            corefs = [coref for coref in self.corefs
                      if sentences[0]['acc_length'] < coref['start']
                      < sentences[-1]['acc_length'] + len(sentences[-1]['sentence'])]

            return self.__class__(tokens,
                                  corefs, speakers, c(self.filename),
                                  c(self.raw_text),  sentences)
        return self

    def speaker(self, i):
        if self.speakers[i[0]] == self.speakers[i[-1]]:
            return self.speakers[i[0]]
        return None
