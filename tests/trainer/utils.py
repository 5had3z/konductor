import torch


def make_dataset(n_samples: int, bias: float = 0.5):
    """Make trivial classification task"""
    assert 0 < bias < 1, "bias must be bewtween 0 and 1"
    n1 = int(n_samples * bias)
    n2 = n_samples - n1
    pop1 = torch.empty(n1).normal_(-1, 2)
    pop2 = torch.empty(n2).normal_(1, 2)
    genpop = torch.cat([pop1, pop2])
    lbl1 = torch.full_like(pop1, 0)
    lbl2 = torch.full_like(pop2, 1)
    genlbl = torch.cat([lbl1, lbl2])
    indices = torch.randperm(n_samples)
    return genpop[indices, None], genlbl[indices, None]


class TrivialLearner(torch.nn.Linear):
    def forward(self, data):
        return super().forward(data[0])


class TrivialLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, pred, tgt):
        return {"bce": super().forward(pred, tgt[1])}
