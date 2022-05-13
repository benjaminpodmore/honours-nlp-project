import torch
import torch.nn as nn
import torch.nn.functional as F
from scorer import Score
import attr
from text import Span
from utils import compute_idx_spans, pad_and_stack, prune


class MentionScorer(nn.Module):
    def __init__(self, input_dim=768, prune_p=0.4):
        super(MentionScorer, self).__init__()

        self.attention_scorer = Score(input_dim)
        self.mention_scorer = Score(3 * input_dim, dropout_p=0.5)
        self.prune_p = prune_p

    def forward(self, embeds, doc, K=250):
        spans = [Span(i1=i[0], i2=i[-1], id=idx, speaker=doc.speaker(i))
                 for idx, i in enumerate(compute_idx_spans(doc.sents, L=10))]

        attns = self.attention_scorer(embeds)

        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2 + 1], embeds[s.i1:s.i2 + 1])
                                        for s in spans])

        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        # Compute weighted attention
        attn_weights = F.softmax(padded_attns, dim=1)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        start_end = torch.stack([torch.cat((embeds[s.i1], embeds[s.i2]))
                                 for s in spans])

        g_i = torch.cat((start_end, attn_embeds), dim=1)

        mention_scores = self.mention_scorer(g_i)

        # Assign mention score to each span
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        # Keep only (n_tokens)*LAMBDA number of spans
        # spans = prune(spans, len(doc), LAMBDA=self.prune_p)

        # spans = [span for span in spans if (span.i1, span.i2) in [a_span, b_span, pronoun_span]]

        # Update possible antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx - K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores


class MentionScorerGap(nn.Module):
    def __init__(self, input_dim=768):
        super(MentionScorerGap, self).__init__()

        self.attention_scorer = Score(input_dim)
        self.mention_scorer = Score(3 * input_dim, dropout_p=0.5)

    # def forward(self, embeds, doc, a_span, b_span, pronoun_span, K=250):
    def forward(self, embeds, doc, a_span, b_span, pronoun_span, K=250):
        spans = [Span(i1=i[0], i2=i[-1], id=idx, speaker=doc.speaker(i))
                 for idx, i in enumerate(compute_idx_spans(doc.sents, L=10))]

        attns = self.attention_scorer(embeds)

        span_attns, span_embeds = zip(*[(attns[s.i1:s.i2 + 1], embeds[s.i1:s.i2 + 1])
                                        for s in spans])

        padded_attns, _ = pad_and_stack(span_attns, value=-1e10)
        padded_embeds, _ = pad_and_stack(span_embeds)

        # Compute weighted attention
        attn_weights = F.softmax(padded_attns, dim=1)
        attn_embeds = torch.sum(torch.mul(padded_embeds, attn_weights), dim=1)

        start_end = torch.stack([torch.cat((embeds[s.i1], embeds[s.i2]))
                                 for s in spans])

        g_i = torch.cat((start_end, attn_embeds), dim=1)

        mention_scores = self.mention_scorer(g_i)

        # Assign mention score to each span
        spans = [
            attr.evolve(span, si=si)
            for span, si in zip(spans, mention_scores.detach())
        ]

        spans = [span for span in spans if (span.i1, span.i2) in [a_span, b_span, pronoun_span]]

        # Update possible antecedents
        spans = [
            attr.evolve(span, yi=spans[max(0, idx - K):idx])
            for idx, span in enumerate(spans)
        ]

        return spans, g_i, mention_scores
