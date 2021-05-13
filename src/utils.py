import torch
from pytorch_lightning.metrics import Metric


class MicroF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("prec_d", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("rec_d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):

        self.n += len(g.intersection(p))
        self.prec_d += len(p)
        self.rec_d += len(g)

    def compute(self):
        p = self.n.float() / self.prec_d
        r = self.n.float() / self.rec_d
        return (2 * p * r / (p + r)) if (p + r) > 0 else (p + r)


class MacroF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):

        prec = len(g.intersection(p)) / len(p)
        rec = len(g.intersection(p)) / len(g)

        self.n += (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else (prec + rec)
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


class MicroPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p))
        self.d += len(p)

    def compute(self):
        return (self.n.float() / self.d) if self.d > 0 else self.d


class MacroPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p)) / len(p)
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


class MicroRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p))
        self.d += len(g)

    def compute(self):
        return (self.n.float() / self.d) if self.d > 0 else self.d


class MacroRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p)) / len(g)
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


def get_markdown(sentences, entity_spans):
    return_outputs = []
    for sent, entities in zip(sentences, entity_spans):
        text = ""
        last_end = 0
        for begin, end, href in entities:
            text += sent[last_end:begin]
            text += "[{}](https://en.wikipedia.org/wiki/{})".format(
                sent[begin:end], href.replace(" ", "_")
            )
            last_end = end

        text += sent[last_end:]
        return_outputs.append(text)

    return return_outputs


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
