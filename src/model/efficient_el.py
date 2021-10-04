import csv
import json
from argparse import ArgumentParser
from collections import defaultdict

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LongformerForMaskedLM,
    get_linear_schedule_with_warmup,
)

from src.data.dataset_el import DatasetEL
from src.model.entity_detection import EntityDetectionFactor
from src.model.entity_linking import EntityLinkingLSTM
from src.utils import (
    MacroF1,
    MacroPrecision,
    MacroRecall,
    MicroF1,
    MicroPrecision,
    MicroRecall,
)


class EfficientEL(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            type=str,
            default="../data/aida_train_dataset.jsonl",
        )
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default="../data/aida_val_dataset.jsonl",
        )
        parser.add_argument(
            "--test_data_path",
            type=str,
            default="../data/aida_test_dataset.jsonl",
        )
        parser.add_argument("--batch_size", type=int, default=2)
        parser.add_argument("--lr_transformer", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--max_length_train", type=int, default=1024)
        parser.add_argument("--max_length", type=int, default=4096)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=10000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--max_length_span", type=int, default=15)
        parser.add_argument("--threshold", type=int, default=0)
        parser.add_argument("--test_with_beam_search", action="store_true")
        parser.add_argument(
            "--test_with_beam_search_no_candidates", action="store_true"
        )
        parser.add_argument(
            "--model_name", type=str, default="allenai/longformer-base-4096"
        )
        parser.add_argument(
            "--mentions_filename",
            type=str,
            default="../data/mentions.json",
        )
        parser.add_argument(
            "--entities_filename",
            type=str,
            default="../data/entities.json",
        )
        parser.add_argument("--epsilon", type=float, default=0.1)
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)

        longformer = LongformerForMaskedLM.from_pretrained(
            self.hparams.model_name,
            num_hidden_layers=8,
            attention_window=[128] * 8,
        )

        self.encoder = longformer.longformer

        self.encoder.embeddings.word_embeddings.weight.requires_grad_(False)

        self.entity_detection = EntityDetectionFactor(
            self.hparams.max_length_span,
            self.hparams.dropout,
            mentions_filename=self.hparams.mentions_filename,
        )

        self.entity_linking = EntityLinkingLSTM(
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.encoder.embeddings.word_embeddings,
            longformer.lm_head,
            self.hparams.dropout,
        )

        self.micro_f1 = MicroF1()
        self.micro_prec = MicroPrecision()
        self.micro_rec = MicroRecall()

        self.macro_f1 = MacroF1()
        self.macro_prec = MacroPrecision()
        self.macro_rec = MacroRecall()

        self.ed_micro_f1 = MicroF1()
        self.ed_micro_prec = MicroPrecision()
        self.ed_micro_rec = MicroRecall()

        self.ed_macro_f1 = MacroF1()
        self.ed_macro_prec = MacroPrecision()
        self.ed_macro_rec = MacroRecall()

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset") or self.hparams.sharded:
            self.train_dataset = DatasetEL(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path,
                max_length=self.hparams.max_length_train,
                max_length_span=self.hparams.max_length_span,
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = DatasetEL(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path,
                max_length=self.hparams.max_length,
                max_length_span=self.hparams.max_length_span,
                test=True,
            )
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            self.test_dataset = DatasetEL(
                tokenizer=self.tokenizer,
                data_path=self.hparams.test_data_path,
                max_length=self.hparams.max_length,
                max_length_span=self.hparams.max_length_span,
                test=True,
            )

        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
        )

    def forward_all_targets(self, batch, return_dict=False):

        hidden_states = self.encoder(
            input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"]
        ).last_hidden_state

        (
            (start, end, scores_ed),
            (
                logits_classifier_start,
                logits_classifier_end,
            ),
        ) = self.entity_detection.forward_hard(
            batch, hidden_states, threshold=self.hparams.threshold
        )

        if start.shape[0] == 0:
            return []

        batch["offsets_start"] = start.T.tolist()
        batch["offsets_end"] = end.T.tolist()

        batch_candidates = [
            {
                (s, e): c
                for (s, e, _), c in zip(
                    batch["raw"][i]["anchors"], batch["raw"][i]["candidates"]
                )
            }.get(tuple((s, e)), ["NIL"])
            for (i, s), (_, e) in zip(
                zip(*batch["offsets_start"]), zip(*batch["offsets_end"])
            )
        ]

        try:
            for k, v in self.tokenizer(
                [c for candidates in batch_candidates for c in candidates],
                return_tensors="pt",
                padding=True,
            ).items():
                batch[f"cand_{k}"] = v.to(self.device)

            batch["offsets_candidates"] = [
                i
                for i, candidates in enumerate(batch_candidates)
                for _ in range(len(candidates))
            ]
            batch["split_candidates"] = [
                len(candidates) for candidates in batch_candidates
            ]

            tokens, scores_el = self.entity_linking.forward_all_targets(
                batch, hidden_states
            )

        except:
            if not self.training:
                print("error on generation")

        try:
            spans = self._tokens_scores_to_spans(batch, start, end, tokens, scores_el)
        except:
            if not self.training:
                print("error on _tokens_scores_to_spans")

            spans = [[[0, 0, [("NIL", 0)]]] for i in range(len(batch["src_input_ids"]))]

        if return_dict:
            return {
                "spans": spans,
                "start": start,
                "end": end,
                "scores_ed": scores_ed,
                "scores_el": scores_el,
                "logits_classifier_start": logits_classifier_start,
                "logits_classifier_end": logits_classifier_end,
            }
        else:
            return spans

    def forward_beam_search(self, batch, candidates=False):

        hidden_states = self.encoder(
            input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"]
        ).last_hidden_state

        (
            (start, end, scores_ed),
            (
                logits_classifier_start,
                logits_classifier_end,
            ),
        ) = self.entity_detection.forward_hard(
            batch, hidden_states, threshold=self.hparams.threshold
        )

        if start.shape[0] == 0:
            return []

        if start.shape[0] == 0:
            return []

        batch["offsets_start"] = start.T.tolist()
        batch["offsets_end"] = end.T.tolist()

        batch_trie_dict = None
        if candidates:
            batch_candidates = [
                {
                    (s, e): c
                    for (s, e, _), c in zip(
                        batch["raw"][i]["anchors"], batch["raw"][i]["candidates"]
                    )
                }.get(tuple((s, e)), ["NIL"])
                for (i, s), (_, e) in zip(
                    zip(*batch["offsets_start"]), zip(*batch["offsets_end"])
                )
            ]

            batch_trie_dict = []
            for candidates in batch_candidates:
                trie_dict = defaultdict(set)
                for c in self.tokenizer(candidates)["input_ids"]:
                    for i in range(1, len(c)):
                        trie_dict[tuple(c[:i])].add(c[i])

                batch_trie_dict.append({k: list(v) for k, v in trie_dict.items()})
        else:
            batch_trie_dict = [self.global_trie] * start.shape[0]

        tokens, scores_el = self.entity_linking.forward_beam_search(
            batch,
            hidden_states,
            batch_trie_dict,
        )

        return self._tokens_scores_to_spans(batch, start, end, tokens, scores_el)

    def _tokens_scores_to_spans(self, batch, start, end, tokens, scores_el):

        spans = [
            [
                [
                    s,
                    e,
                    list(
                        zip(
                            self.tokenizer.batch_decode(t, skip_special_tokens=True),
                            l.tolist(),
                        )
                    ),
                ]
                for s, e, t, l in zip(
                    start[start[:, 0] == i][:, 1].tolist(),
                    end[end[:, 0] == i][:, 1].tolist(),
                    tokens[start[:, 0] == i],
                    scores_el[start[:, 0] == i],
                )
            ]
            for i in range(len(batch["src_input_ids"]))
        ]

        for spans_ in spans:
            for e in [
                [x, y]
                for x in spans_
                for y in spans_
                if x is not y and x[1] >= y[0] and x[0] <= y[0]
            ]:
                for x in sorted(e, key=lambda x: x[1] - x[0])[:-1]:
                    spans_.remove(x)

        return spans

    def training_step(self, batch, batch_idx=None):

        hidden_states = self.encoder(
            input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"]
        ).last_hidden_state

        loss_start, loss_end = self.entity_detection.forward_loss(batch, hidden_states)

        loss_generation, loss_classifier = self.entity_linking.forward_loss(
            batch, hidden_states, epsilon=self.hparams.epsilon
        )

        self.log("loss_s", loss_start, on_step=True, on_epoch=False, prog_bar=True)
        self.log("loss_e", loss_end, on_step=True, on_epoch=False, prog_bar=True)
        self.log("loss_g", loss_generation, on_step=True, on_epoch=False, prog_bar=True)
        self.log("loss_c", loss_classifier, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss_start + loss_end + loss_generation + loss_classifier}

    def _inference_step(self, batch, batch_idx=None):
        if self.hparams.test_with_beam_search_no_candidates:
            spans = self.forward_beam_search(batch)
        elif self.hparams.test_with_beam_search:
            spans = self.forward_beam_search(batch, candidates=True)
        else:
            spans = self.forward_all_targets(batch)

        for p, g in zip(spans, batch["raw"]):

            p_ = set((e[0], e[1], e[2][0][0]) for e in p)
            g_ = set((e[0], e[1], e[2]) for e in g["anchors"])

            self.micro_f1(p_, g_)
            self.micro_prec(p_, g_)
            self.micro_rec(p_, g_)

            self.macro_f1(p_, g_)
            self.macro_prec(p_, g_)
            self.macro_rec(p_, g_)

            p_ = set((e[0], e[1]) for e in p)
            g_ = set((e[0], e[1]) for e in g["anchors"])

            self.ed_micro_f1(p_, g_)
            self.ed_micro_prec(p_, g_)
            self.ed_micro_rec(p_, g_)

            self.ed_macro_f1(p_, g_)
            self.ed_macro_prec(p_, g_)
            self.ed_macro_rec(p_, g_)

        return {
            "micro_f1": self.micro_f1,
            "micro_prec": self.micro_prec,
            "macro_rec": self.macro_rec,
            "macro_f1": self.macro_f1,
            "macro_prec": self.macro_prec,
            "micro_rec": self.micro_rec,
            "ed_micro_f1": self.ed_micro_f1,
            "ed_micro_prec": self.ed_micro_prec,
            "ed_micro_rec": self.ed_micro_rec,
            "ed_macro_f1": self.ed_macro_f1,
            "ed_macro_prec": self.ed_macro_prec,
            "ed_macro_rec": self.ed_macro_rec,
        }

    def validation_step(self, batch, batch_idx=None):
        metrics = self._inference_step(batch, batch_idx)
        self.log_dict(
            {k: v for k, v in metrics.items() if k in ("micro_f1", "ed_micro_f1")},
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx=None):
        metrics = self._inference_step(batch, batch_idx)
        self.log_dict(metrics)

    def generate_global_trie(self):

        with open(self.hparams.entities_filename) as f:
            entities = json.load(f)

        trie_dict = defaultdict(set)
        for e in tqdm(entities, desc="Loading .."):
            c = self.tokenizer(e)["input_ids"]
            for i in range(1, len(c)):
                trie_dict[tuple(c[:i])].add(c[i])

        self.global_trie = {k: list(v) for k, v in trie_dict.items()}

    def sample(self, sentences, anchors=None, candidates=None, all_targets=False):
        self.eval()
        with torch.no_grad():
            batch = {
                f"src_{k}": v.to(self.device)
                for k, v in self.tokenizer(
                    sentences,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.hparams.max_length,
                    truncation=True,
                ).items()
            }

            batch["raw"] = [
                {
                    "input": sentence,
                    "anchors": anchors[i] if anchors else None,
                    "candidates": candidates[i] if candidates else None,
                }
                for i, sentence in enumerate(sentences)
            ]

            if anchors and candidates and all_targets:
                spans = self.forward_all_targets(batch)
            elif candidates and all_targets:
                spans = self.forward_beam_search(batch, candidates=True)
            else:
                spans = self.forward_beam_search(batch)

            return [
                [(eo[s][0].item(), eo[e][1].item(), l) for s, e, l in es]
                for es, eo in zip(spans, batch["src_offset_mapping"])
            ]

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.encoder.named_parameters()
                    if "embbedding" not in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.lr_transformer,
            },
            {
                "params": [
                    p
                    for n, p in self.encoder.named_parameters()
                    if "embbedding" not in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.hparams.lr_transformer,
            },
            {
                "params": [
                    p
                    for n, p in self.entity_detection.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.entity_detection.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
            },
            {
                "params": [
                    p
                    for n, p in self.entity_linking.named_parameters()
                    if "embedding" not in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.entity_linking.named_parameters()
                    if "embbedding" not in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", "frequency": 1}],
        )
