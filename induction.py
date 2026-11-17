"""Helpers for reproducing induction heads in GPT-2 small (Olsson et al., 2022).

Sequence layout used everywhere: [BOS, x_1 .. x_L, x_1 .. x_L], length 2L + 1.
Position 0 is BOS, the first copy is 1..L, the second copy is L+1..2L.

`loss[:, j]` below is the loss for predicting token j+1 from position j.
We score the first copy on j = 1..L-1 and the second copy on j = L+1..2L-1:
both are L-1 predictions, and in the second copy every one of them can be
solved by induction (the previous token appeared earlier, followed by the answer).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

# NLTK UDHR file ids. Articles 1-25 are present in all four versions.
LANGS = {
    "en": "English-Latin1",
    "es": "Spanish-Latin1",
    "de": "German_Deutsch-Latin1",
    "sv": "Swedish_Svenska-Latin1",
}
LANG_NAMES = {"en": "English", "es": "Spanish", "de": "German", "sv": "Swedish"}
ARTICLE_MARKER = {
    "en": r"Article (\d+)",
    "es": r"Art[íi]culo (\d+)",
    "de": r"Artikel (\d+)",
    "sv": r"Artikel (\d+)",
}
N_SHARED_ARTICLES = 25


def load_model(name: str = "gpt2-small") -> HookedTransformer:
    model = HookedTransformer.from_pretrained(name, device="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------- inputs

def repeat(model: HookedTransformer, seq: torch.Tensor) -> torch.Tensor:
    """[batch, L] tokens -> [batch, 2L+1] with BOS + seq + seq."""
    bos = torch.full((seq.shape[0], 1), model.tokenizer.bos_token_id, dtype=seq.dtype)
    return torch.cat([bos, seq, seq], dim=1)


def random_tokens(model: HookedTransformer, L: int, batch: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    seq = torch.randint(0, model.cfg.d_vocab, (batch, L), generator=g)
    return repeat(model, seq)


def load_udhr(lang: str) -> str:
    import nltk
    from nltk.corpus import udhr

    nltk.download("udhr", quiet=True)
    return udhr.raw(LANGS[lang])


def clean(text: str) -> str:
    return " ".join(text.split())


def udhr_articles(lang: str, n: int = N_SHARED_ARTICLES) -> list[str]:
    """Articles 1..n of the UDHR, as whitespace-normalised strings (marker removed)."""
    raw = load_udhr(lang)
    parts = re.split(ARTICLE_MARKER[lang], raw)
    # re.split with a group gives [preamble, num, body, num, body, ...]
    articles = {int(num): clean(body).lstrip(".: ") for num, body in zip(parts[1::2], parts[2::2])}
    return [articles[i] for i in range(1, n + 1)]


def tokenize(model: HookedTransformer, text: str) -> torch.Tensor:
    """1-D token ids for `text` with a leading space (so the first word looks like any other).

    Uses the raw tokenizer: model.to_tokens would silently truncate at n_ctx = 1024.
    """
    ids = model.tokenizer.encode(" " + text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)


def token_windows(model, text: str, L: int, n: int) -> torch.Tensor:
    """n non-overlapping windows of exactly L tokens from `text`, each repeated. -> [n, 2L+1]."""
    toks = tokenize(model, text)
    n_avail = len(toks) // L
    if n_avail < n:
        raise ValueError(f"only {n_avail} windows of {L} tokens available, asked for {n}")
    seq = toks[: n * L].reshape(n, L)
    return repeat(model, seq)


def shuffle_words(text: str, seed: int = 0) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


# ---------------------------------------------------------------- tokenizer stats

def is_word_initial(model, token_ids: torch.Tensor) -> torch.Tensor:
    """True where the token starts a new word (GPT-2 marks this with a leading space)."""
    strs = model.to_str_tokens(token_ids.flatten(), prepend_bos=False)
    flags = torch.tensor([s.startswith(" ") for s in strs])
    return flags.reshape(token_ids.shape)


def tokenizer_stats(model, text: str) -> dict:
    toks = tokenize(model, text)
    n_words = len(text.split())
    initial = is_word_initial(model, toks)
    return {
        "words": n_words,
        "tokens": len(toks),
        "tokens_per_word": len(toks) / n_words,
        "continuation_share": 1 - initial.float().mean().item(),
    }


# ---------------------------------------------------------------- metrics

def per_token_loss(model, tokens: torch.Tensor, fwd_hooks=()) -> torch.Tensor:
    """[batch, 2L] loss for predicting token j+1 from position j."""
    logits = model.run_with_hooks(tokens, fwd_hooks=list(fwd_hooks))
    logprobs = logits[:, :-1].log_softmax(-1)
    return -logprobs.gather(-1, tokens[:, 1:, None])[..., 0]


def halves(L: int) -> tuple[slice, slice]:
    """Index slices into per_token_loss output for first and second copy."""
    return slice(1, L), slice(L + 1, 2 * L)


@dataclass
class LossSummary:
    first: float
    second: float

    @property
    def gain(self) -> float:
        """In-context gain: how much easier the second copy is than the first."""
        return self.first - self.second


def loss_summary(loss: torch.Tensor, L: int, mask: torch.Tensor | None = None) -> LossSummary:
    """Mean first/second-copy loss. `mask` ([batch, 2L], over targets) restricts which targets count."""
    a, b = halves(L)
    if mask is None:
        return LossSummary(loss[:, a].mean().item(), loss[:, b].mean().item())
    return LossSummary(loss[:, a][mask[:, a]].mean().item(), loss[:, b][mask[:, b]].mean().item())


def target_word_initial_mask(model, tokens: torch.Tensor) -> torch.Tensor:
    """[batch, 2L] mask aligned with per_token_loss: True if the *target* token is word-initial."""
    return is_word_initial(model, tokens[:, 1:])


def attention_patterns(model, tokens: torch.Tensor) -> torch.Tensor:
    """[layer, batch, head, q, k] attention patterns."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    return torch.stack([cache["pattern", l] for l in range(model.cfg.n_layers)])


def induction_scores(patterns: torch.Tensor, L: int) -> torch.Tensor:
    """[layer, head] mean attention from q (second copy) to q - L + 1.

    q - L + 1 is the token right after the earlier occurrence of the token at q.
    """
    diag = patterns.diagonal(offset=-(L - 1), dim1=-2, dim2=-1)  # element i = pattern[i+L-1, i]
    return diag[..., 2:].mean(dim=(1, -1))  # i >= 2  <=>  q >= L+1


def prev_token_scores(patterns: torch.Tensor) -> torch.Tensor:
    """[layer, head] mean attention from q to q - 1 (over q >= 1)."""
    return patterns.diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=(1, -1))


def top_heads(scores: torch.Tensor, k: int) -> list[tuple[int, int]]:
    n_heads = scores.shape[1]
    idx = scores.flatten().argsort(descending=True)[:k]
    return [(int(i) // n_heads, int(i) % n_heads) for i in idx]


def fmt_heads(heads) -> str:
    return ", ".join(f"{l}.{h}" for l, h in heads)


# ---------------------------------------------------------------- ablation
#
# Mean ablation replaces a head's output (hook_z) with its average output. The average is
# computed on a *reference* batch that is never the batch being evaluated: a held-out batch
# from the same distribution, or (for short texts) the other half of the windows, see
# `crossfit_ablated_loss`. Zero ablation sets the head's output to 0 and needs no reference.

def head_means(model, reference: torch.Tensor) -> torch.Tensor:
    """[layer, head, d_head] mean of every head's output (hook_z) over batch and position of `reference`."""
    names = {f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)}
    _, cache = model.run_with_cache(reference, names_filter=lambda n: n in names)
    return torch.stack([cache["z", l].mean(dim=(0, 1)) for l in range(model.cfg.n_layers)])


def ablation_hooks(heads, means: torch.Tensor | None = None) -> list:
    """Hooks that replace each head's output with `means[l, h]`, or with 0 if `means` is None."""
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    hooks = []
    for l, hs in by_layer.items():
        def hook(z, hook, l=l, hs=hs):
            z[:, :, hs] = 0.0 if means is None else means[l, hs]
            return z

        hooks.append((f"blocks.{l}.attn.hook_z", hook))
    return hooks


def ablated_loss(model, tokens, heads, means: torch.Tensor | None = None) -> torch.Tensor:
    """[batch, 2L] per-token loss with `heads` ablated (mean ablation with `means`, zero ablation if None)."""
    return per_token_loss(model, tokens, ablation_hooks(heads, means))


def two_folds(n: int) -> list[torch.Tensor]:
    """Interleaved halves of n sequences: even and odd indices."""
    return [torch.arange(0, n, 2), torch.arange(1, n, 2)]


def fold_means(model, tokens: torch.Tensor) -> list[torch.Tensor]:
    """Head means computed separately on each of the two folds of `tokens`."""
    return [head_means(model, tokens[idx]) for idx in two_folds(len(tokens))]


def crossfit_ablated_loss(model, tokens, heads, folds_means: list[torch.Tensor]) -> torch.Tensor:
    """Mean-ablated per-token loss where each fold is ablated with the means of the *other* fold.

    For texts too short to hold out a separate batch: every sequence is still evaluated with means
    computed on sequences it is not part of.
    """
    loss = torch.empty(tokens.shape[0], tokens.shape[1] - 1)
    for k, idx in enumerate(two_folds(len(tokens))):
        loss[idx] = ablated_loss(model, tokens[idx], heads, folds_means[1 - k])
    return loss


def random_heads(model, k: int, exclude, rng: np.random.Generator) -> list[tuple[int, int]]:
    pool = [
        (l, h)
        for l in range(model.cfg.n_layers)
        for h in range(model.cfg.n_heads)
        if (l, h) not in set(exclude)
    ]
    return [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]


def gain_removed(clean: LossSummary, ablated: LossSummary) -> float:
    """Share of the clean in-context gain that disappears under ablation."""
    return (clean.gain - ablated.gain) / clean.gain


# ---------------------------------------------------------------- natural-text diagnostics

def any_target_induction_scores(patterns: torch.Tensor, tokens: torch.Tensor, L: int) -> torch.Tensor:
    """[layer, head] attention from q (second copy) to *any* earlier key k whose previous token equals token q.

    In natural text a token can occur several times, so several keys are valid induction targets
    and the strict score (only q - L + 1) undercounts.
    """
    B, T = tokens.shape
    q = torch.arange(L + 1, 2 * L + 1)
    prev = torch.cat([torch.full((B, 1), -1), tokens[:, :-1]], dim=1)  # prev[b, k] = tokens[b, k-1]
    valid = prev[:, None, :].eq(tokens[:, q][:, :, None])              # [B, nq, T]
    valid &= torch.arange(T)[None, None, :] < q[None, :, None]
    valid[..., 0] = False
    return (patterns[:, :, :, q, :] * valid[None, :, None]).sum(-1).mean(dim=(1, -1))


def ambiguous_share(tokens: torch.Tensor, L: int) -> float:
    """Share of second-copy positions whose token occurs more than once in the first copy."""
    first, second = tokens[:, 1 : L + 1], tokens[:, L + 1 : 2 * L + 1]
    counts = second[:, :, None].eq(first[:, None, :]).sum(-1)
    return (counts > 1).float().mean().item()
