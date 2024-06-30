import jax
import lz4.frame
import optax
from cloudpickle import cloudpickle


def get_optimizer(lr, b1, b2, eps, weight_decay, warmup_pct, n_steps, div_factor, final_div_factor, lr_schedule="constant"):
    if lr_schedule == "cosine":
        lr = optax.cosine_onecycle_schedule(n_steps, lr, warmup_pct, div_factor, final_div_factor)
    elif lr_schedule == "constant":
        lr = optax.constant_schedule(lr)

    return optax.adamw(lr, b1, b2, eps, weight_decay=weight_decay), lr


def gradient_step(variables, loss_params, opt_state, optimizer, loss_fn):
    params = variables.pop('params')
    state = variables

    (loss, state), grads = jax.value_and_grad(lambda p: loss_fn({'params': p, **state}, *loss_params), has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return {'params': params, **state}, opt_state, loss


def init(model, key, *x, print_summary=False):
    params_key, gpt_key, dropout_key = jax.random.split(key, 3)
    variables = model.init({'params': params_key, 'gpt': gpt_key, 'dropout': dropout_key}, *x)

    if print_summary:
        print(model.tabulate(jax.random.key(0), *x, compute_flops=True))

    return variables


def init_cache(model, *x):
    variables = model.init({'params': jax.random.PRNGKey(0)}, *x, training=False)
    return variables['cache']


def forward(model, variables, key, *x, method=None):
    gpt_key, dropout_key = jax.random.split(key)
    return model.apply(variables, *x, rngs={'gpt': gpt_key, 'dropout': dropout_key}, mutable=list(set(variables) - {'params'}), method=method)


def save_variables(*variables, path):
    with lz4.frame.open(path, 'wb') as f:
        cloudpickle.dump(variables, f)


def load_variables(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.load(f)
