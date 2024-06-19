import jax
import lz4.frame
import optax
from cloudpickle import cloudpickle


def gradient_step(params, loss_params, opt_state, optimizer, loss_fn):
    grads, aux = jax.grad(loss_fn, has_aux=True)(params, *loss_params)
    updates, opt_state = optimizer.update(grads, opt_state, params=params, grad_fn=jax.grad(lambda p, _: loss_fn(p, *loss_params)[0]))
    params = optax.apply_updates(params, updates)

    return params, opt_state, aux


def init(model, key, *x, print_summary=False):
    params_key, gpt_key, dropout_key = jax.random.split(key, 3)

    variables = model.init({'params': params_key, 'gpt': gpt_key, 'dropout': dropout_key}, *x)
    params = variables.pop('params')
    state = variables

    if print_summary:
        print(model.tabulate(jax.random.key(0), *x, compute_flops=True))

    return params, state


def forward(model, params, state, key, *x, method=None):
    gpt_key, dropout_key = jax.random.split(key)
    return model.apply({'params': params, **state}, *x, rngs={'gpt': gpt_key, 'dropout': dropout_key}, mutable=list(state.keys()), method=method)


def save_model(params, state, path):
    with lz4.frame.open(path, 'wb') as f:
        cloudpickle.dump((params, state), f)


def load_model(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.load(f)
