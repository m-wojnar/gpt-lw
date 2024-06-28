from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from nltk.util import ngrams

from gpt_lw.model_utils import forward


def compute_relative_positions(tokens, delim_token):
    batch_size, seq_len = tokens.shape
    relative_positions = jnp.full_like(tokens, fill_value=-seq_len)
    
    delim_mask = (tokens == delim_token) 
    delim_mask = delim_mask.at[:, 0].set(True) # sometimes first pos is not delim
    
    positions = jnp.arange(seq_len).reshape(1, -1)
    relative_positions = jnp.where(delim_mask, positions, relative_positions)

    # Compute cumulative maximum along the sequence length for each batch
    def cumulative_max_scan(carry, x):
        carry = jnp.maximum(carry, x)
        return carry, carry

    last_pos = jax.lax.scan(cumulative_max_scan, relative_positions[:, 0], relative_positions.T)[1].T
    # Compute relative positions
    relative_positions = positions - last_pos
    return relative_positions


def compute_ngram_weights(train_tokens, val_tokens, ngram_sizes):
    train_tokens = train_tokens.tolist()
    val_tokens = val_tokens.tolist()
    n = len(train_tokens)

    train_weights = np.zeros_like(train_tokens, dtype=np.float32)
    val_weights = np.zeros_like(val_tokens, dtype=np.float32)

    for size in ngram_sizes:
        ng = ngrams(train_tokens, size)
        ng = Counter(ng)
        w = jax.tree.map(lambda x: x / (n - size + 1), dict(ng))
        w = jax.tree.map(lambda x: 1 - x, w)
        norm = sum(w[k] * ng[k] for k in ng) / (n - size + 1)
        w = jax.tree.map(lambda x: x / norm, w)

        train_weights[:size - 1] += 1
        val_weights[:size - 1] += 1

        for i in range(size - 1, len(train_tokens)):
            train_weights[i] += w[tuple(train_tokens[i - size + 1:i + 1])]

        for i in range(size - 1, len(val_tokens)):
            val_weights[i] += w[tuple(val_tokens[i - size + 1:i + 1])]

    train_weights /= len(ngram_sizes)
    val_weights /= len(ngram_sizes)

    return jnp.asarray(train_weights), jnp.asarray(val_weights)


def get_weighted_loss(model, weighting, precomputed):
    if weighting == "unweighted":
        def unweighted(x):
            return 1.0
        weight_fn = unweighted
    elif weighting == "negexp_relpos":
        def negexp_relpos(x):
            relative_positions = compute_relative_positions(x, delim_token=0)
            weights = (-jnp.exp(-relative_positions) + 1.0) * (27.0 / 26.0) + 1e-3
            return weights
        weight_fn = negexp_relpos

    def weighted_nt(variables, key, xt, xtp1, weights):
        logits, state = forward(model, variables, key, xt)
        token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, xtp1)

        if not precomputed:
            weights = weight_fn(xt)

        weighted_loss = token_loss * weights
        return weighted_loss.mean(), state

    return weighted_nt