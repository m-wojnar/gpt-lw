import jax
import jax.numpy as jnp

import itertools

# NOTE: this OOMs
# def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
#     def _loss_fn(variables, key, xt_row, xtp1_row, token_pos):
#         loss, _ = loss_fn(variables, key, xt_row, xtp1_row)
#         return loss[:, token_pos].mean()  # Ensure this is a scalar
# 
#     params = variables.pop('params')
#     state = variables
# 
#     batch_size, seq_length = xt.shape
# 
#     def compute_loss_and_grads(xt_row, xtp1_row):
#         xt_row = xt_row[jnp.newaxis, :]  # Ensure correct shape (1, seq_length)
#         xtp1_row = xtp1_row[jnp.newaxis, :]  # Ensure correct shape (1, seq_length)
#         
#         # Vectorize the computation of loss and gradients for the current row
#         loss_row, grads_row = jax.vmap(
#             jax.value_and_grad(lambda p, j: _loss_fn({'params': p, **state}, key, xt_row, xtp1_row, j)),
#             in_axes=(None, 0)
#         )(params, jnp.arange(seq_length))
# 
#         # Compute the gradient norms for the current row
#         grad_norms_row = jax.tree_map(lambda x: (x ** 2).reshape(x.shape[0], -1).sum(axis=1), grads_row)
#         grad_norms_row = jnp.asarray(jax.tree_leaves(grad_norms_row)).sum(axis=0)
# 
#         return loss_row, jnp.sqrt(grad_norms_row)
# 
#     # Use vmap to apply the function over the batch dimension
#     loss_container, grad_norms_container = jax.vmap(compute_loss_and_grads, in_axes=(0, 0))(xt, xtp1)
# 
#     print(loss_container.shape, grad_norms_container.shape)
# 
#     return loss_container, grad_norms_container

def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
    def _loss_fn(variables, key, xt_row, xtp1_row, token_pos):
        loss, _ = loss_fn(variables, key, xt_row, xtp1_row)
        return loss[0, token_pos]

    params = variables.pop('params')
    state = variables

    batch_size, seq_length = xt.shape

    # Initialize containers for losses and grad norms
    loss_container = jnp.zeros((batch_size, seq_length))
    grad_norms_container = jnp.zeros((batch_size, seq_length))

    for i in range(batch_size):
        xt_row = xt[i:i+1]
        xtp1_row = xtp1[i:i+1]
        
        # Vectorize the computation of loss and gradients for the current row
        loss_row, grads_row = jax.vmap(
            jax.value_and_grad(lambda p, j: _loss_fn({'params': p, **state}, key, xt_row, xtp1_row, j)),
            in_axes=(None, 0)
        )(params, jnp.arange(seq_length))

        # Store the loss
        loss_container = loss_container.at[i, :].set(loss_row)

        # Compute the gradient norms for the current row
        grad_norms_row = jax.tree_map(lambda x: (x ** 2).reshape(x.shape[0], -1).sum(axis=1), grads_row)
        grad_norms_row = jnp.asarray(jax.tree_leaves(grad_norms_row)).sum(axis=0)

        # Store the gradient norms
        grad_norms_container = grad_norms_container.at[i, :].set(jnp.sqrt(grad_norms_row))

    print(loss_container.shape, grad_norms_container.shape)

    return loss_container, grad_norms_container