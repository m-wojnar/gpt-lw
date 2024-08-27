import jax
import jax.numpy as jnp


def grad_tree_to_grad_norm(grads, sum_axis):
    grad_norms = jax.tree_map(lambda x: (x ** 2).sum(axis=tuple(range(sum_axis, x.ndim))), grads)
    grad_norms = jnp.asarray(jax.tree_leaves(grad_norms)).sum(axis=0).T
    grad_norms = jnp.sqrt(grad_norms)
    return grad_norms


# NOTE: Computes mean grad norm for each token in the input sequence
def grad_norm_per_token(loss_fn, points, variables, key, xt, xtp1):
    params = variables['params']
    state = {k: v for k, v in variables.items() if k != 'params'}

    grads = jax.vmap(
        jax.grad(lambda p, i: loss_fn({'params': p, **state}, key, xt, xtp1)[0][:, i].mean()),
        in_axes=(None, 0)
    )(params, points)
    grad_norms = grad_tree_to_grad_norm(grads, sum_axis=1)

    return grad_norms


def grad_norm(loss_fn, variables, key, xt, xtp1):
    params = variables['params']
    state = {k: v for k, v in variables.items() if k != 'params'}

    grads, _ = jax.grad(loss_fn, has_aux=True)({'params': params, **state}, key, xt, xtp1)
    grad_norm = grad_tree_to_grad_norm(grads, sum_axis=0)

    return grad_norm