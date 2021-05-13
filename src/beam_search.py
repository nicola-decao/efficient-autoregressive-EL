"""
Adapted from: https://github.com/probabll/bayeseq/blob/master/aevnmt/components/beamsearch.py
"""
import torch
import torch.nn.functional as F
from packaging import version


# from onmt
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        return [tile(e, count, dim=dim) for e in x]

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def beam_search(
    decoder,
    tgt_vocab_size,
    hidden,
    bos_idx,
    eos_idx,
    pad_idx,
    beam_width,
    alpha=1,
    max_len=15,
    batch_trie_dict=None,
):
    """
    Beam search with size beam_width. Follows OpenNMT-py implementation.
    In each decoding step, find the k most likely partial hypotheses.

    :param decoder: an initialized decoder
    """

    decoder.eval()
    with torch.no_grad():

        # Initialize the hidden state and create the initial input.
        batch_size = (
            hidden[0].shape[0] if isinstance(hidden, tuple) else hidden.shape[0]
        )
        device = hidden[0].device if isinstance(hidden, tuple) else hidden.device

        prev_y = torch.full(
            size=[batch_size],
            fill_value=bos_idx,
            dtype=torch.long,
            device=device,
        )

        # Tile hidden decoder states and encoder outputs beam_width times
        hidden = tile(hidden, beam_width, dim=0)  # [layers, B*beam_width, H_dec]

        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_width,
            step=beam_width,
            dtype=torch.long,
            device=device,
        )
        alive_seq = torch.full(
            [batch_size * beam_width, 1],
            bos_idx,
            dtype=torch.long,
            device=device,
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (beam_width - 1), device=device
        ).repeat(batch_size)

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["gold_score"] = [0] * batch_size
        results["contexts"] = [[] for _ in range(batch_size)]

        done = torch.full(
            [batch_size, beam_width],
            False,
            dtype=torch.bool,
            device=device,
        )
        trie_idx = (
            torch.arange(0, batch_size, device=device)
            .unsqueeze(-1)
            .repeat(1, beam_width)
            .view(-1)
        )
        for step in range(max_len):
            prev_y = alive_seq[:, -1].view(-1)

            # expand current hypotheses, decode one single step
            log_probs, hidden = decoder.step_beam_search(prev_y, hidden)

            if batch_trie_dict is not None:
                mask = torch.full_like(log_probs, -float("inf"))
                for i, (b_idx, tokens) in enumerate(
                    zip(trie_idx.tolist(), alive_seq.tolist())
                ):
                    idx = batch_trie_dict[b_idx].get(tuple(tokens), [])
                    mask[[i] * len(idx), idx] = 0

                log_probs += mask

            # multiply probs by the beam probability (=add logprobs)
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            curr_scores = log_probs

            # compute length penalty
            if alpha > -1:
                length_penalty = (step + 1) ** alpha
                curr_scores /= length_penalty

            # flatten log_probs into a list of possibilities
            curr_scores = curr_scores.reshape(-1, beam_width * tgt_vocab_size)

            # pick currently best top beam_width hypotheses (flattened order)
            topk_scores, topk_ids = curr_scores.topk(beam_width, dim=-1)

            if alpha > -1:
                # recover original log probs
                topk_log_probs = topk_scores * length_penalty

            # reconstruct beam origin and true word ids from flattened order
            if version.parse(torch.__version__) >= version.parse("1.5.0"):
                topk_beam_index = topk_ids.floor_divide(tgt_vocab_size)
            else:
                topk_beam_index = topk_ids.div(tgt_vocab_size)
            topk_ids = topk_ids.fmod(tgt_vocab_size)

            # map beam_index to batch_index in the flat representation
            batch_index = topk_beam_index + beam_offset[
                : topk_beam_index.size(0)
            ].unsqueeze(1)
            select_indices = batch_index.view(-1)

            # append latest prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
            )  # batch_size*k x hyp_len

            is_finished = (
                topk_ids.eq(eos_idx) & ~topk_scores.eq(-float("inf"))
            ) | topk_scores.eq(-float("inf")).all(-1, keepdim=True)

            if step + 1 == max_len:
                is_finished.fill_(1)

            done |= is_finished

            # end condition is whether the top beam is finished
            end_condition = done.all(-1)

            # for LSTMs, states are tuples of tensors
            hidden = [e.index_select(0, select_indices) for e in hidden]
            trie_idx = trie_idx.index_select(0, select_indices)

            # save finished hypotheses
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_width, alive_seq.size(-1))
                contexts = hidden[1].view(-1, beam_width, hidden[1].shape[-1])

                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    finished_hyp = is_finished[i].nonzero(as_tuple=False).view(-1)

                    # store finished hypotheses for this batch
                    for j in finished_hyp:

                        hypotheses[b].append(
                            (
                                topk_scores[i, j],
                                predictions[i, j],
                                contexts[i, j].clone(),
                            )  # ignore start_token
                        )
                    # if the batch reached the end, save the beam_width hypotheses
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True
                        )
                        for n, (score, pred, cont) in enumerate(best_hyp):
                            if n >= beam_width:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                            results["contexts"][b].append(cont)

            if end_condition.any():
                non_finished = end_condition.eq(0).nonzero(as_tuple=False).view(-1)

                # if all sentences are translated, no need to go further
                if len(non_finished) == 0:
                    break

                # remove finished batches for the next step
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(
                    -1, alive_seq.size(-1)
                )
                done = done.index_select(0, non_finished)

                # reorder indices, outputs and masks, and trie
                select_indices = batch_index.view(-1)

                # for LSTMs, states are tuples of tensors
                hidden = [e.index_select(0, select_indices) for e in hidden]
                trie_idx = trie_idx.index_select(0, select_indices)

    return (
        torch.nn.utils.rnn.pad_sequence(
            [
                torch.nn.utils.rnn.pad_sequence(
                    e, batch_first=True, padding_value=pad_idx
                ).T
                for e in results["predictions"]
            ],
            batch_first=True,
            padding_value=pad_idx,
        ).permute(0, 2, 1),
        torch.stack([torch.stack(e) for e in results["scores"]]),
        torch.stack([torch.stack(e) for e in results["contexts"]]),
    )
