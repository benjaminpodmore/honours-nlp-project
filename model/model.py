import torch
import torch.nn as nn
from document import DocumentEncoder
from mention import MentionScorer, MentionScorerGap
from linker import MentionLinkScorer
from utils import to_cuda


class CorefScorer(nn.Module):
    def __init__(self, tokenizer=None, model=None):
        super().__init__()
        self.encoder = to_cuda(DocumentEncoder(tokenizer, model))
        self.mention_scorer = to_cuda(MentionScorer(prune_p=0.4))
        self.mention_link_scorer = to_cuda(MentionLinkScorer())

    def forward(self, doc):
        embeds = self.encoder(doc)
        spans, g_i, mention_scores = self.mention_scorer(embeds, doc, K=250)
        spans, coref_scores = self.mention_link_scorer(spans, g_i, mention_scores)
        return spans, coref_scores


class CorefScorerGap(nn.Module):
    def __init__(self, tokenizer=None, model=None):
        super().__init__()
        self.encoder = to_cuda(DocumentEncoder(tokenizer, model))
        self.mention_scorer = to_cuda(MentionScorerGap())
        self.mention_link_scorer = to_cuda(MentionLinkScorer())

    def forward(self, doc, a_span, b_span, pronoun_span):
        embeds = self.encoder(doc)
        spans, g_i, mention_scores = self.mention_scorer(embeds, doc, a_span, b_span, pronoun_span, K=250)
        spans, coref_scores = self.mention_link_scorer(spans, g_i, mention_scores)
        # return 3
        # return spans, coref_scores

