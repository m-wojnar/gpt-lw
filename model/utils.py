import jax
import jax.numpy as jnp


def batches(*x, batch_size, shuffle_key=None):
    n = len(x[0])

    if shuffle_key is not None:
        perm = jax.random.permutation(shuffle_key, jnp.arange(n))
        x = tuple(x_i[perm] for x_i in x)

    for i in range(0, n, batch_size):
        yield tuple(x_i[i:i + batch_size] for x_i in x)
