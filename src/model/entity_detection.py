import json

import torch


class EntityDetectionFactor(torch.nn.Module):
    def __init__(self, max_length_span, dropout=0, mentions_filename=None):
        super().__init__()

        self.max_length_span = max_length_span
        self.classifier_start = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1),
        )
        self.classifier_end = torch.nn.Sequential(
            torch.nn.LayerNorm(768 * 2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(768 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1),
        )

        self.mentions = None
        if mentions_filename:
            with open(mentions_filename) as f:
                self.mentions = set(json.load(f))

    def _forward_start(self, batch, hidden_states):
        return self.classifier_start(hidden_states).squeeze(-1)

    def _forward_end(self, batch, hidden_states, offsets_start):

        classifier_end_input = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, self.max_length_span - 1)
        )

        classifier_end_input = torch.cat(
            (
                hidden_states[offsets_start]
                .unsqueeze(-1)
                .repeat(1, 1, self.max_length_span),
                classifier_end_input.unfold(1, self.max_length_span, 1)[offsets_start],
            ),
            dim=1,
        ).permute(0, 2, 1)

        logits_classifier_end = self.classifier_end(classifier_end_input).squeeze(-1)

        mask = torch.cat(
            (
                batch["src_attention_mask"],
                torch.zeros(
                    (
                        batch["src_attention_mask"].shape[0],
                        self.max_length_span - 1,
                    ),
                    dtype=torch.float,
                    device=hidden_states.device,
                ),
            ),
            dim=1,
        )
        mask = torch.where(
            mask.bool(),
            torch.zeros_like(mask),
            -torch.full_like(mask, float("inf")),
        ).unfold(1, self.max_length_span, 1)[offsets_start]

        return logits_classifier_end + mask

    def forward(self, batch, hidden_states):

        logits_classifier_start = self._forward_start(batch, hidden_states)
        offsets_start = batch["offsets_start"]
        logits_classifier_end = self._forward_end(batch, hidden_states, offsets_start)

        return logits_classifier_start, logits_classifier_end

    def forward_hard(self, batch, hidden_states, threshold=0):

        logits_classifier_start = self._forward_start(batch, hidden_states)
        offsets_start = logits_classifier_start > threshold
        logits_classifier_end = self._forward_end(batch, hidden_states, offsets_start)

        start = offsets_start.nonzero()
        end = start.clone()

        scores = None
        if logits_classifier_end.shape[0] > 0:
            end[:, 1] += logits_classifier_end.argmax(-1)
            scores = (
                logits_classifier_start[offsets_start]
                + logits_classifier_end.max(-1).values
            )
            if self.mentions:
                mention_mask = torch.tensor(
                    [
                        (
                            batch["raw"][i]["input"][
                                batch["src_offset_mapping"][i][s][0]
                                .item() : batch["src_offset_mapping"][i][e][1]
                                .item()
                            ]
                            in self.mentions
                        )
                        for (i, s), (_, e) in zip(start, end)
                    ],
                    device=start.device,
                )

                start = start[mention_mask]
                end = end[mention_mask]

        return (start, end, scores), (
            logits_classifier_start,
            logits_classifier_end,
        )

    def forward_loss(self, batch, hidden_states):
        logits_classifier_start, logits_classifier_end = self.forward(
            batch, hidden_states
        )

        batch["labels_start"] = torch.zeros_like(batch["src_input_ids"])
        batch["labels_start"][batch["offsets_start"]] = 1

        loss_start = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_classifier_start,
            batch["labels_start"].float(),
            weight=batch["src_attention_mask"],
        )

        batch["labels_end"] = torch.tensor(
            [b - a for a, b in zip(batch["offsets_start"][1], batch["offsets_end"][1])],
            device=logits_classifier_start.device,
        )

        loss_end = torch.nn.functional.cross_entropy(
            logits_classifier_end,
            batch["labels_end"],
        )

        return loss_start, loss_end
