from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.optimize import minimize
from tqdm import tqdm


def loss_fn(params, X, Y, i):
    logits = X @ params
    return optax.softmax_cross_entropy(logits[None], Y).mean()


def grad_norm(params, X, Y):
    grads = jax.vmap(
        jax.grad(loss_fn), in_axes=(None, 0, 0, 0)
    )(params, X, Y, jnp.arange(len(X)))
    return jnp.sqrt((grads ** 2).mean(axis=0).sum(axis=1))


def sample_batch(params, key, bs, n_cat, n_entropy_points):
    x_key, y_key = jax.random.split(key, 2)
    X = jax.random.categorical(x_key, jnp.zeros((bs, n_entropy_points)), axis=-1)
    X = jax.nn.one_hot(X, num_classes=n_entropy_points)
    Y = X @ params
    Y = jax.random.categorical(y_key, Y, axis=-1)
    Y = jax.nn.one_hot(Y, num_classes=n_cat, dtype=int)
    return X, Y


def create_trained_model(n_cat, n_entropy_points):
    entropy = np.linspace(0, np.log2(n_cat), n_entropy_points)
    params = []

    def _entropy(probs, H):
        probs = np.asarray(probs)
        probs /= probs.sum()
        return jnp.abs(-(probs * np.log2(probs)).sum() - H)

    for H in entropy:
        probs = minimize(_entropy, np.full(n_cat, 1 / n_cat), (H,), method='Powell', bounds=[(0., 1.) for _ in range(n_cat)]).x
        params.append(probs)

    params = np.asarray(params)
    params = np.log(params)
    return params


if __name__ == '__main__':
    n_cat = 20
    n_entropy_points = 20
    n_steps = 2500
    n_bs = 16

    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    params = create_trained_model(n_cat, n_entropy_points)

    batch_sizes = np.logspace(0, n_bs - 1, n_bs,  base=2).astype(int)
    entropy = np.linspace(0, np.log2(n_cat), n_entropy_points)
    results = []

    for bs in tqdm(batch_sizes):
        bs_results = []
        sample_fn = jax.jit(partial(sample_batch, bs=bs, n_cat=n_cat, n_entropy_points=n_entropy_points))
        grad_fn = jax.jit(grad_norm)

        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            X, Y = sample_fn(params, subkey)
            bs_results.append(grad_fn(params, X, Y))

        results.append(np.asarray(bs_results).mean(axis=0))

    plt.imshow(results)
    plt.yticks(range(n_bs), batch_sizes)
    plt.xticks(range(n_entropy_points), [f'{e:.2f}' for e in entropy], rotation=90)
    plt.ylabel('Batch size')
    plt.xlabel('Entropy')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('simple_model.pdf', bbox_inches='tight')
    plt.show()
