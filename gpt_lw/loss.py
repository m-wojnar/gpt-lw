import os
import jax
import jax.numpy as jnp
import optax

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


def get_weighted_loss(model, weighting, delim_token=-1):
    if weighting == "unweighted":
        def unweighted(x):
            return 1.0
        weight_fn = unweighted
    elif weighting == "negexp_relpos":
        def negexp_relpos(x):
            relative_positions = compute_relative_positions(x, delim_token=delim_token)
            weights = (-jnp.exp(-relative_positions) + 1.0) * (27.0 / 26.0) + 1e-3
            return weights
        weight_fn = negexp_relpos
    elif os.path.exists(weighting): # passed through tensor
        weights = jnp.load(weighting)
        def tensor_relpos(x):
            relative_positions = compute_relative_positions(x, delim_token=delim_token)
            return weights[relative_positions]
        weight_fn = tensor_relpos

    def weighted_nt(variables, key, xt, xtp1):
        logits, state = forward(model, variables, key, xt)
        token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, xtp1)
        weights = weight_fn(xt)
        print(weights)

        weighted_loss = token_loss * weights
        return weighted_loss.mean(), state

    return weighted_nt

def get_per_token_loss(model):
    def per_token_loss(variables, key, xt, xtp1):
        logits, state = forward(model, variables, key, xt)
        token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, xtp1)
        return token_loss, state

    return per_token_loss