import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from boltons.iterutils import windowed
from itertools import combinations, tee


def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]


def compute_idx_spans(sentences, L=10):
    """ Compute span indexes for all possible spans up to length L in each
    sentence """
    idx_spans, shift = [], 0
    for sent in sentences:
        sent_spans = flatten([windowed(range(shift, len(sent)+shift), length)
                              for length in range(1, L+1)])
        idx_spans.extend(sent_spans)
        shift += len(sent)

    return idx_spans


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def get_word_embedding(encoded, output, idx):
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded, sizes


def prune(spans, T, LAMBDA=0.40):
    """ Prune mention scores to the top lambda percent.
    Returns list of tuple(scores, indices, g_i) """

    # Only take top λT spans, where T = len(doc)
    STOP = int(LAMBDA * T)

    # Sort by mention score, remove overlapping spans, prune to top λT spans
    sorted_spans = sorted(spans, key=lambda s: s.si, reverse=True)
    nonoverlapping = remove_overlapping(sorted_spans)
    pruned_spans = nonoverlapping[:STOP]

    # Resort by start, end indexes
    spans = sorted(pruned_spans, key=lambda s: (s.i1, s.i2))

    return spans


def remove_overlapping(sorted_spans):
    """ Remove spans that are overlapping by order of decreasing mention score
    unless the current span i yields true to the following condition with any
    previously accepted span j:

    si.i1 < sj.i1 <= si.i2 < sj.i2   OR
    sj.i1 < si.i1 <= sj.i2 < si.i2 """

    # Nonoverlapping will be accepted spans, seen is start, end indexes that
    # have already been seen in an accepted span
    nonoverlapping, seen = [], set()
    for s in sorted_spans:
        indexes = range(s.i1, s.i2+1)
        taken = [i in seen for i in indexes]
        if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
            nonoverlapping.append(s)
            seen.update(indexes)

    return nonoverlapping


def pairwise_indexes(spans):
    """ Get indices for indexing into pairwise_scores """
    indexes = [0] + [len(s.yi) for s in spans]
    indexes = [sum(indexes[:idx+1]) for idx, _ in enumerate(indexes)]
    return bert_pairwise(indexes)


def bert_pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return x.requires_grad_()


def speaker_label(s1, s2):
    """ Compute if two spans have the same speaker or not """
    # Same speaker
    if s1.speaker == s2.speaker:
        idx = torch.tensor(1)

    # Different speakers
    elif s1.speaker != s2.speaker:
        idx = torch.tensor(2)

    # No speaker
    else:
        idx = torch.tensor(0)

    return idx


def to_cuda(x):
    """ GPU-enable a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def safe_divide(x, y):
    """ Make sure we don't divide by 0 """
    if y != 0:
        return x / y
    return 1


def extract_gold_corefs(document):
    """ Parse coreference dictionary of a document to get coref links """

    # Initialize defaultdict for keeping track of corefs
    gold_links = defaultdict(list)

    # Compute number of mentions
    gold_mentions = set([coref['span'] for coref in document.corefs])
    total_mentions = len(gold_mentions)

    # Compute number of coreferences
    for coref_entry in document.corefs:

        # Parse label of coref span, the span itself
        label, span_idx = coref_entry['label'], coref_entry['span']

        # All spans corresponding to the same label
        gold_links[label].append(span_idx) # get all spans corresponding to some label

    # Flatten all possible corefs, sort, get number
    gold_corefs = flatten([[coref
                            for coref in combinations(gold, 2)]
                            for gold in gold_links.values()])
    gold_corefs = sorted(gold_corefs)
    total_corefs = len(gold_corefs)

    return gold_corefs, total_corefs, gold_mentions, total_mentions

def bs(list_, target_):
    lo, hi = 0, len(list_) -1

    while lo < hi:
        mid = lo + int((hi - lo) / 2)

        if target_ < list_[mid]:
            hi = mid
        elif target_ > list_[mid]:
            lo = mid + 1
        else:
            return mid + 1
    return lo

def bs_(list_, target_):
    lo, hi = 0, len(list_) -1

    while lo < hi:
        mid = lo + int((hi - lo) / 2)

        if target_ < list_[mid]:
            hi = mid
        elif target_ > list_[mid]:
            lo = mid + 1
        else:
            return mid
    return lo