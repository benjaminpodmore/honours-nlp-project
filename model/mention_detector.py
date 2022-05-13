import torch
import torch.nn as nn
from document import DocumentEncoder
from mention import MentionScorer
from linker import MentionLinkScorer
from utils import to_cuda


class MentionDetector(nn.Module):
    def __init__(self, tokenizer=None, model=None):
        super().__init__()
        self.encoder = to_cuda(DocumentEncoder(tokenizer, model))
        self.mention_scorer = to_cuda(MentionScorer())
        self.mention_link_scorer = to_cuda(MentionLinkScorer())

    def forward(self, doc):
        embeds = self.encoder(doc)
        spans, g_i, mention_scores = self.mention_scorer(embeds, doc, K=250)
        return spans, mention_scores

