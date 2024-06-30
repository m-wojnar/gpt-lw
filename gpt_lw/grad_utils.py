import jax
import jax.numpy as jnp

import itertools


def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
    def _loss_fn(variables, key, xt, xtp1, batch_pos, token_pos):
        loss, _ = loss_fn(variables, key, xt, xtp1)
        return loss[batch_pos, token_pos]

    params = variables.pop('params')
    state = variables

    # Generate pairs of (i, j) for each token in each batch
    batch_size, seq_length = xt.shape
    batch_indices, token_indices = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(seq_length), indexing='ij')
    
    # Flatten the indices to pass them to vmap
    batch_indices = batch_indices.flatten()
    token_indices = token_indices.flatten()

    # Vectorize the computation of loss and gradients
    loss, grads = jax.vmap(
        jax.value_and_grad(lambda p, i, j: _loss_fn({'params': p, **state}, key, xt, xtp1, i, j)),
        in_axes=(None, 0, 0)
    )(params, batch_indices, token_indices)

    # Reshape the loss to match the input shape
    loss = loss.reshape((batch_size, seq_length))

    # Compute the gradient norms
    grad_norms = jax.tree_map(lambda x: (x ** 2).reshape(x.shape[0], -1).sum(axis=1), grads)
    grad_norms = jnp.asarray(jax.tree_leaves(grad_norms)).sum(axis=0)

    # Reshape the gradient norms to match the input shape
    grad_norms = grad_norms.reshape((batch_size, seq_length))
    grad_norms = jnp.sqrt(grad_norms)
    return loss, grad_norms