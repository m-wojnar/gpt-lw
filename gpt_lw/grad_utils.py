import jax
import jax.numpy as jnp


def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
    params = variables.pop('params')
    state = variables

    jacobian = jax.vmap(
        jax.jacrev(lambda p, x, y: loss_fn({'params': p, **state}, key, x[None], y[None])[0][0]),
        in_axes=(None, 0, 0)
    )(params, xt, xtp1)
    loss, _ = loss_fn({'params': params, **state}, key, xt, xtp1)

    grad_norms = jax.tree.map(lambda x: (x ** 2).sum(axis=tuple(range(2, x.ndim))), jacobian)
    grad_norms = jnp.asarray(jax.tree.leaves(grad_norms)).sum(axis=0)
    grad_norms = jnp.sqrt(grad_norms)

    # fake grad norms
    # grad_norms = jnp.zeros_like(loss)

    return loss, grad_norms

