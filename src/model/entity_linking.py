import torch

from src.beam_search import beam_search
from src.utils import label_smoothed_nll_loss


class LSTM(torch.nn.Module):
    def __init__(
        self, bos_token_id, pad_token_id, eos_token_id, embeddings, lm_head, dropout=0
    ):
        super().__init__()

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.embeddings = embeddings
        self.lm_head = lm_head
        self.dropout = dropout

        self.lstm_cell = torch.nn.LSTMCell(
            input_size=2 * 768,
            hidden_size=768,
        )

    def _roll(
        self,
        input_ids,
        attention_mask,
        decoder_hidden,
        decoder_context,
        decoder_append,
        return_lprob=False,
        return_dict=False,
    ):

        dropout_mask = 1
        if self.training:
            dropout_mask = (torch.rand_like(decoder_hidden) > self.dropout).float()

        all_hiddens = []
        all_contexts = []

        emb = self.embeddings(input_ids)
        for t in range(emb.shape[1]):
            decoder_hidden, decoder_context = self.lstm_cell(
                torch.cat((emb[:, t], decoder_append), dim=-1),
                (decoder_hidden, decoder_context),
            )
            decoder_hidden *= dropout_mask
            all_hiddens.append(decoder_hidden)
            all_contexts.append(decoder_context)

        all_hiddens = torch.stack(all_hiddens, dim=1)
        all_contexts = torch.stack(all_contexts, dim=1)

        all_contexts = all_contexts[
            [e for e in range(attention_mask.shape[0])], attention_mask.sum(-1) - 1
        ]

        if return_dict:
            outputs = {
                "all_hiddens": all_hiddens,
                "all_contexts": all_contexts,
            }
        else:
            outputs = (all_hiddens, all_contexts)
        if return_lprob:
            logits = self.lm_head(all_hiddens)

            if self.training:
                logits = logits.log_softmax(-1)
            else:
                logits.sub_(logits.max(-1, keepdim=True).values)
                logits.exp_()
                logits.div_(logits.sum(-1, keepdim=True))
                logits.log_()

            if return_dict:
                outputs = {
                    "logits": logits,
                }
            else:
                outputs += (logits,)

        return outputs

    def step_beam_search(self, previous_tokens, hidden):
        decoder_hidden, decoder_context, decoder_append = hidden

        emb = self.embeddings(previous_tokens)
        decoder_hidden, decoder_context = self.lstm_cell(
            torch.cat((emb, decoder_append), dim=-1),
            (decoder_hidden, decoder_context),
        )

        logits = self.lm_head(decoder_hidden)
        logits.sub_(logits.max(-1, keepdim=True).values)
        logits.exp_()
        logits.div_(logits.sum(-1, keepdim=True))
        logits.log_()

        return logits, (decoder_hidden, decoder_context, decoder_append)

    def forward(self, batch, decoder_hidden, decoder_context, decoder_append):

        _, all_contexts_positive, lprobs = self._roll(
            batch["trg_input_ids"][:, :-1],
            batch["trg_attention_mask"][:, 1:],
            decoder_hidden,
            decoder_context,
            decoder_append,
            return_lprob=True,
        )

        _, all_contexts_negative = self._roll(
            batch["neg_input_ids"][:, :-1],
            batch["neg_attention_mask"][:, 1:],
            decoder_hidden[batch["neg_mask"]],
            decoder_context[batch["neg_mask"]],
            decoder_append[batch["neg_mask"]],
            return_lprob=False,
        )

        return all_contexts_positive, all_contexts_negative, lprobs

    def forward_all_targets(
        self, batch, decoder_hidden, decoder_context, decoder_append
    ):

        _, all_contexts, lprobs = self._roll(
            batch["cand_input_ids"][:, :-1],
            batch["cand_attention_mask"][:, 1:],
            decoder_hidden,
            decoder_context,
            decoder_append,
            return_lprob=True,
        )

        scores = (
            lprobs.gather(
                dim=-1, index=batch["cand_input_ids"][:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            * batch["cand_attention_mask"][:, 1:]
        )

        return all_contexts, scores.sum(-1) / (batch["cand_attention_mask"].sum(-1) - 1)

    def forward_beam_search(self, batch, hidden_states):
        raise NotImplemented


class EntityLinkingLSTM(torch.nn.Module):
    def __init__(
        self, bos_token_id, pad_token_id, eos_token_id, embeddings, lm_head, dropout=0
    ):
        super().__init__()

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.prj = torch.nn.Sequential(
            torch.nn.LayerNorm(768 * 2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(768 * 2, 768),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(768, 768 * 3),
        )

        self.lstm = LSTM(
            bos_token_id,
            pad_token_id,
            eos_token_id,
            embeddings,
            lm_head,
            dropout=dropout,
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(768 * 2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(768 * 2, 768),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(768, 1),
        )

    def _get_hidden_context_append_vectors(self, batch, hidden_states):
        return self.prj(
            torch.cat(
                (
                    hidden_states[batch["offsets_start"]],
                    hidden_states[batch["offsets_end"]],
                ),
                dim=-1,
            )
        ).split([768, 768, 768], dim=-1)

    def forward(self, batch, hidden_states):
        (
            decoder_hidden,
            decoder_context,
            decoder_append,
        ) = self._get_hidden_context_append_vectors(batch, hidden_states)

        (
            all_contexts_positive,
            all_contexts_negative,
            lprobs_lm,
        ) = self.lstm.forward(batch, decoder_hidden, decoder_context, decoder_append)

        logits_classifier = self.classifier(
            torch.cat(
                (
                    decoder_append[batch["neg_mask"]].unsqueeze(1).repeat(1, 2, 1),
                    torch.stack(
                        (
                            all_contexts_positive[batch["neg_mask"]],
                            all_contexts_negative,
                        ),
                        dim=1,
                    ),
                ),
                dim=-1,
            )
        ).squeeze(-1)

        return lprobs_lm, logits_classifier

    def forward_loss(self, batch, hidden_states, epsilon=0):

        lprobs_lm, logits_classifier = self.forward(batch, hidden_states)

        loss_generation, _ = label_smoothed_nll_loss(
            lprobs_lm,
            batch["trg_input_ids"][:, 1:],
            epsilon=epsilon,
            ignore_index=self.pad_token_id,
        )
        loss_generation = loss_generation / batch["trg_attention_mask"][:, 1:].sum()

        loss_classifier = torch.nn.functional.cross_entropy(
            logits_classifier,
            torch.zeros(
                (logits_classifier.shape[0]),
                dtype=torch.long,
                device=logits_classifier.device,
            ),
        )

        return loss_generation, loss_classifier

    def forward_all_targets(self, batch, hidden_states):
        (
            decoder_hidden,
            decoder_context,
            decoder_append,
        ) = self._get_hidden_context_append_vectors(batch, hidden_states)

        all_contexts, lm_scores = self.lstm.forward_all_targets(
            batch,
            decoder_hidden[batch["offsets_candidates"]],
            decoder_context[batch["offsets_candidates"]],
            decoder_append[batch["offsets_candidates"]],
        )

        classifier_scores = self.classifier(
            torch.cat(
                (
                    decoder_append[batch["offsets_candidates"]],
                    all_contexts,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        scores = lm_scores + classifier_scores

        classifier_scores = torch.cat(
            [
                e.log_softmax(-1)
                for e in classifier_scores.split(batch["split_candidates"])
            ]
        )

        tokens = [
            t[s.argsort(descending=True)]
            for s, t in zip(
                scores.split(batch["split_candidates"]),
                batch["cand_input_ids"].split(batch["split_candidates"]),
            )
        ]

        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=self.pad_token_id
        )

        scores = torch.nn.utils.rnn.pad_sequence(
            [
                e.sort(descending=True).values
                for e in scores.split(batch["split_candidates"])
            ],
            batch_first=True,
            padding_value=-float("inf"),
        )

        return tokens, scores

    def forward_beam_search(
        self, batch, hidden_states, batch_trie_dict=None, beams=5, alpha=1, max_len=15
    ):
        (
            decoder_hidden,
            decoder_context,
            decoder_append,
        ) = self._get_hidden_context_append_vectors(batch, hidden_states)

        tokens, lm_scores, all_contexts = beam_search(
            self.lstm,
            self.lstm.lm_head.decoder.out_features,
            (decoder_hidden, decoder_context, decoder_append),
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            beam_width=beams,
            alpha=alpha,
            max_len=max_len,
            batch_trie_dict=batch_trie_dict,
        )

        classifier_scores = self.classifier(
            torch.cat(
                (
                    decoder_append.unsqueeze(1).repeat(1, beams, 1),
                    all_contexts,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        classifier_scores[lm_scores == -float("inf")] = -float("inf")
        classifier_scores = classifier_scores.log_softmax(-1)

        scores = (classifier_scores + lm_scores).sort(-1, descending=True)
        tokens = tokens[
            torch.arange(scores.indices.shape[0], device=tokens.device)
            .unsqueeze(-1)
            .repeat(1, beams),
            scores.indices,
        ]
        scores = scores.values

        return tokens, scores
