import os
import jax
import jax.numpy as jnp
import optax

from gpt_lw.model_utils import forward


def compute_relative_positions(tokens, delim_token):
    batch_size, seq_len = tokens.shape
    relative_positions = jnp.ones((batch_size, seq_len), dtype=jnp.int32) * -seq_len
    
    delim_mask = (tokens == delim_token) 
    delim_mask = delim_mask.at[:, 0].set(True) # default
    
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


def get_weighted_loss(model, weighting, delim_token=-1):
    if weighting == "unweighted":
        def unweighted(x):
            return jnp.ones_like(x)
        weight_fn = unweighted
    elif weighting == "negexp_relpos":
        def negexp_relpos(x):
            relative_positions = compute_relative_positions(x, delim_token=delim_token)
            weights = (-jnp.exp(-relative_positions) + 1.0) * (27.0 / 26.0) + 1e-3
            return weights
        weight_fn = negexp_relpos
    elif os.path.exists(weighting): # passed through tensor
        weights = jnp.load(weighting)
        *_, name = weighting.split("/")
        def tensor_abspos(x):
            if 'rel' in name:
                relative_positions = compute_relative_positions(x, delim_token=delim_token)
                return weights[relative_positions]
            elif 'abs' in name:
                absolute_positions = jnp.arange(x.shape[1]).reshape(1, -1).repeat(x.shape[0], axis=0)
                return weights[absolute_positions]
            else:
                raise ValueError(f"Invalid weighting name: {name}")
        weight_fn = tensor_abspos

    def weighted_nt(variables, key, xt, xtp1):
        logits, state = forward(model, variables, key, xt)
        token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, xtp1)

        weights = weight_fn(xt)
        weights *= xt.shape[1] / weights.sum(axis=1, keepdims=True)

        weighted_loss = token_loss * weights
        return weighted_loss, state

    return weighted_nt
