import torch
import torch.nn as nn
import torch.nn.functional as F
import attr
from scorer import Score
from speaker import Speaker
from utils import pairwise_indexes, speaker_label, to_cuda, to_var


class MentionLinkScorer(nn.Module):
    def __init__(self, embed_dim=768, speaker_dim=20):
        super(MentionLinkScorer, self).__init__()
        self.speaker = Speaker(speaker_dim)
        self.score = Score(3 * 3 * embed_dim + speaker_dim, dropout_p=0.5)

    def forward(self, spans, g_i, mention_scores):
        mention_ids, antecedent_ids, speaker_labels = zip(*[(i.id, j.id, speaker_label(i, j))
                                                            for i in spans
                                                            for j in i.yi])

        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))
        speaker_labels = to_cuda(torch.tensor(speaker_labels))

        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)
        phi = self.speaker(speaker_labels)

        pairs = torch.cat((i_g, j_g, i_g * j_g, phi), dim=1)

        s_ij = self.score(pairs)

        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)
        spans = [
            attr.evolve(span,
                        yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi]
                        )
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        split_scores = [to_cuda(torch.tensor([]))] + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(to_cuda(torch.tensor([[0.]])))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        scores = list(torch.split(coref_scores, antecedent_idx, dim=0))

        with_epsilon_probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
        # probs = [F.softmax(tensr, dim=0) for tensr in scores]

        return spans, with_epsilon_probs
