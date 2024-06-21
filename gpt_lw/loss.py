import optax
import jax.numpy as jnp
from jax import lax
from optax import losses
from flax import linen as nn

# TODO: needs benchmarking, can we make it faster?
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

    last_pos = lax.scan(cumulative_max_scan, relative_positions[:, 0], relative_positions.T)[1].T
    # Compute relative positions
    relative_positions = positions - last_pos
    return relative_positions


def get_weighted_loss(state, weighting):
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

    def weighted_nt(params, xt, xtp1):
        logits = state.apply_fn({'params': params}, xt)
        token_loss = losses.softmax_cross_entropy_with_integer_labels(logits, xtp1)
        weights = weight_fn(xt)

        weighted_loss = token_loss * weights
        return weighted_loss.mean()

    return weighted_nt