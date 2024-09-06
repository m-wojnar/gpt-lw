from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.optimize import minimize
from tqdm import tqdm


def loss_fn(params, X, Y):
    logits = X @ params
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()


def grad_norm(params, X, Y):
    grads = jax.grad(loss_fn)(params, X, Y)
    return jnp.linalg.norm(grads)


@partial(jax.jit, static_argnames=('bs', 'n_entropy_points'))
def batched_grad_norm(params, logits, i, bs, n_entropy_points, keys):
    X = jnp.zeros((bs, n_entropy_points)).at[:, i].set(1)

    def _fn(key):
        Y = jax.random.categorical(key, logits, shape=(bs,))
        return grad_norm(params, X, Y)

    return jax.vmap(_fn)(keys)


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
    plt.figure(figsize=(8, 3), dpi=300)

    n_cat = 5
    n_entropy_points = 40
    n_samples = 10000
    n_bs = 14

    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    params = create_trained_model(n_cat, n_entropy_points)

    batch_sizes = np.logspace(0, n_bs - 1, n_bs,  base=2).astype(int)
    entropy = np.linspace(0, np.log2(n_cat), n_entropy_points)
    results = []

    for bs in tqdm(batch_sizes):
        bs_results = jnp.zeros(n_entropy_points)

        for i, logits in enumerate(params):
            e_results = batched_grad_norm(params, logits, i, bs, n_entropy_points, jax.random.split(key, max(20, n_samples // bs)))
            bs_results = bs_results.at[i].set(e_results.mean())

        results.append(bs_results)

    plt.imshow(results, cmap='gnuplot2')
    plt.yticks(range(n_bs), batch_sizes)
    plt.xticks(range(n_entropy_points), [f'{e:.2f}' for e in entropy], rotation=90)
    plt.ylabel('Batch size')
    plt.xlabel('Entropy')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('simple_model.pdf', bbox_inches='tight')
    plt.show()
