"""
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
"""
# XIA.py

from torch import nn
from torch.nn import Module


class XIA_multi(Module):
    def __init__(self, nb_att=2, *args, **kwargs):
        super(XIA_multi, self).__init__()
        self.xia_blocs = nn.ModuleList([XIA(*args, **kwargs) for _ in range(nb_att)])

    def forward(self, k1, k2):
        for xia in self.xia_blocs:
            k1 = xia(k1, k2)
        return k1


class XIA(Module):
    def __init__(self, embed_dim=256, nb_h=8, dropout=0.1):
        super(XIA, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nb_h, dropout=dropout)

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, k1, k2):
        # return k1_new
        x = k1.permute(2, 0, 1)
        x = x + self.self_attn(k2.permute(2, 0, 1), x, value=x)[0]
        return self.fc(x).permute(1, 2, 0)
