import torch
import torch.nn as nn
from utils import init_weights


class Score(nn.Module):
    def __init__(self, input_dim, hidden_dim=150, dropout_p=0.10):
        super(Score, self).__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )

        self.score.apply(init_weights)

    def forward(self, x):
        return self.score(x)
