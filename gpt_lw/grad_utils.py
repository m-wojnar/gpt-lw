import jax
import jax.numpy as jnp


def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
    params = variables.pop('params')
    state = variables

    jacobian, _ = jax.jacrev(lambda p: loss_fn({'params': p, **state}, key, xt, xtp1), has_aux=True)(params)
    loss, _ = loss_fn({'params': params, **state}, key, xt, xtp1)

    grad_norms = jax.tree.map(lambda x: (x ** 2).sum(axis=tuple(range(2, x.ndim))), jacobian)
    grad_norms = jnp.asarray(jax.tree.leaves(grad_norms)).sum(axis=0)
    return loss, jnp.sqrt(grad_norms)
