import jax
import jax.numpy as jnp


# NOTE: Computes the grad norm for each token in the input sequence sequentially
def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
    params = variables.pop('params')
    state = variables

    jacobian_fn = jax.jacrev(lambda p, i: jax.lax.dynamic_slice(loss_fn({'params': p, **state}, key, xt, xtp1)[0], (0, i), (xt.shape[0], 1)))
    _, grad_norms = jax.lax.scan(lambda _, i: (None, jacobian_fn(params, i)), None, jnp.arange(xt.shape[1]))
    grad_norms = jax.tree.map(lambda x: (x ** 2).sum(axis=tuple(range(2, x.ndim))), grad_norms)
    grad_norms = jnp.asarray(jax.tree.leaves(grad_norms)).sum(axis=0)
    grad_norms = jnp.sqrt(grad_norms)

    loss, _ = loss_fn({'params': params, **state}, key, xt, xtp1)
    return loss, grad_norms