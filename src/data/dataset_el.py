import os

import jsonlines
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetEL(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=32,
        max_length_span=15,
        test=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        with jsonlines.open(data_path) as f:
            self.data = list(f)

        self.max_length = max_length
        self.max_length_span = max_length_span
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):

        batch = {
            **{
                f"src_{k}": v
                for k, v in self.tokenizer(
                    [b["input"] for b in batch],
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                    return_offsets_mapping=True,
                ).items()
            },
            "offsets_start": (
                [
                    i
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
                [
                    a[0]
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
            ),
            "offsets_end": (
                [
                    i
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
                [
                    a[1]
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                ],
            ),
            "offsets_inside": (
                [
                    i
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                    for j in range(a[0] + 1, a[1] + 1)
                ],
                [
                    j
                    for i, b in enumerate(batch)
                    for a in b["anchors"]
                    if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
                    for j in range(a[0] + 1, a[1] + 1)
                ],
            ),
            "raw": batch,
        }

        if not self.test:

            negatives = [
                np.random.choice([e for e in cands if e != a[2]])
                if len([e for e in cands if e != a[2]]) > 0
                else None
                for b in batch["raw"]
                for a, cands in zip(b["anchors"], b["candidates"])
                if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
            ]

            targets = [
                a[2]
                for b in batch["raw"]
                for a in b["anchors"]
                if a[1] < self.max_length and a[1] - a[0] < self.max_length_span
            ]

            assert len(targets) == len(negatives)

            batch_upd = {
                **(
                    {
                        f"trg_{k}": v
                        for k, v in self.tokenizer(
                            targets,
                            return_tensors="pt",
                            padding=True,
                            max_length=self.max_length,
                            truncation=True,
                        ).items()
                    }
                    if not self.test
                    else {}
                ),
                **(
                    {
                        f"neg_{k}": v
                        for k, v in self.tokenizer(
                            [e for e in negatives if e],
                            return_tensors="pt",
                            padding=True,
                            max_length=self.max_length,
                            truncation=True,
                        ).items()
                    }
                    if not self.test
                    else {}
                ),
                "neg_mask": torch.tensor([e is not None for e in negatives]),
            }

            batch = {**batch, **batch_upd}

        return batch
