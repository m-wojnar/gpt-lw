import jax
import jax.numpy as jnp


def grad_norm_per_token(loss_fn, variables, key, xt, xtp1):
    def _loss_fn(variables, key, xt, xtp1, token_pos):
        loss, _ = loss_fn(variables, key, xt, xtp1)
        return loss[:, token_pos].mean()

    params = variables.pop('params')
    state = variables

    grads = jax.vmap(
        jax.grad(lambda p, i: _loss_fn({'params': p, **state}, key, xt, xtp1, i)),
        in_axes=(None, 0)
    )(params, jnp.arange(xt.shape[1]))

    grad_norms = jax.tree.map(lambda x: (x ** 2).reshape(x.shape[0], -1).sum(axis=1), grads)
    grad_norms = jnp.asarray(jax.tree.leaves(grad_norms)).sum(axis=0)
    return jnp.sqrt(grad_norms)
