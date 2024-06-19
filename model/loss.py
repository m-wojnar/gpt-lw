import optax
from flax import linen as nn

from .nn import forward


def get_weight_fn(dataset):
    def weight_fn(y):
        return 1.0

    return weight_fn


def eval_fn(x, y, weight_fn):
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1)
    y = nn.one_hot(y, x.shape[-1])
    loss = optax.sigmoid_binary_cross_entropy(x, y).reshape(x.shape[0], -1).sum(axis=-1).mean()
    wloss = weight_fn(y) * loss
    return wloss


def loss_fn(params, state, key, x, y, model, weight_fn):
    logits, state = forward(model, params, state, key, x)
    loss = eval_fn(logits, y, weight_fn)
    return loss, (state, loss)
