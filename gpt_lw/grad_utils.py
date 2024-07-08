import jax
import jax.numpy as jnp


def grad_tree_to_grad_norm(grads, sum_axis):
    grad_norms = jax.tree_map(lambda x: (x ** 2).sum(axis=tuple(range(sum_axis, x.ndim))), grads)
    grad_norms = jnp.asarray(jax.tree_leaves(grad_norms)).sum(axis=0).T
    grad_norms = jnp.sqrt(grad_norms)
    return grad_norms


# NOTE: Computes the grad norm for each token in the input sequence sequentially
def grad_norm_per_token(loss_fn, slice_size, variables, key, xt, xtp1):
    params = variables['params']
    state = {k: v for k, v in variables.items() if k != 'params'}

    jacobian_fn = jax.jacrev(lambda p, i: jax.lax.dynamic_slice(loss_fn({'params': p, **state}, key, xt, xtp1)[0], (0, i), (slice_size, 1)))
    _, grads = jax.lax.scan(lambda _, i: (None, jacobian_fn(params, i)), None, jnp.arange(xt.shape[1]))
    grad_norms = grad_tree_to_grad_norm(grads, sum_axis=2)

    return grad_norms


def grad_norm(loss_fn, variables, key, xt, xtp1):
    params = variables['params']
    state = {k: v for k, v in variables.items() if k != 'params'}

    grads, _ = jax.grad(loss_fn, has_aux=True)({'params': params, **state}, key, xt, xtp1)
    grad_norm = grad_tree_to_grad_norm(grads, sum_axis=0)

    return grad_norm