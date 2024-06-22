import os
import yaml
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import optax
from chex import Array

from gpt_lw.data import Tokenizer, get_dataset, sample_batch
from gpt_lw.gpt import GPT, GPTConfig
from gpt_lw.loss import get_weighted_loss
from gpt_lw.nn import get_optimizer, init, gradient_step, save_model


def train(
        run_name: str,
        config: GPTConfig,
        dataset: Array,
        tokenizer: Tokenizer,
        optimizer: optax.GradientTransformation,
        batch_size: int,
        n_steps: int,
        loss_weighting: str,
        seed: int,
        save_freq: int,
        val_freq: int
    ):

    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)

    # gen random keys
    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    # check device
    device = jax.devices()[0].platform
    print("Device: ", device)

    # init model
    model = GPT(config)
    inputs = jnp.empty((batch_size, config.seq_len), dtype=int)
    variables = init(model, init_key, inputs, print_summary=True)

    # init optimizer
    opt_state = optimizer.init(variables["params"])

    # gradient step and eval functions
    loss_fn = get_weighted_loss(model, loss_weighting)
    step_fn = jax.jit(partial(gradient_step, loss_fn=loss_fn, optimizer=optimizer))
    eval_fn = jax.jit(loss_fn)

    # train loop
    for step in range(n_steps):
        train_key, batch_key = jax.random.split(train_key)
        xt, xtp1 = sample_batch(dataset, batch_size, config.seq_len, batch_key)

        variables, opt_state, loss = step_fn(variables, (train_key, xt, xtp1), opt_state)
        print("loss: ", loss)

        if step % val_freq == 0:
            val_key, batch_key = jax.random.split(val_key)
            xt, xtp1 = sample_batch(dataset, batch_size, config.seq_len, batch_key)

            val_loss, _ = eval_fn(variables, val_key, xt, xtp1)
            print("val_loss: ", val_loss)

        if step % save_freq == 0:
            save_model(variables, opt_state, f"checkpoints/{run_name}/step_{step}.pkl")

    # save final model
    save_model(variables, opt_state, f"checkpoints/{run_name}/final.pkl")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--config_path", type=str, default="configs/base.yaml")
    args.add_argument("--dataset_path", type=str, default="cfg_dataset/simple4_100000.txt")
    args.add_argument("--run_name", type=str, default="base")
    args = args.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    dataset, tokenizer = get_dataset(args.dataset_path)
    gpt_config = GPTConfig(**config["gpt_config"])
    optimizer = get_optimizer(**config["optimizer_config"])
    train_config = config["train_config"]

    train(args.run_name, gpt_config, dataset, tokenizer, optimizer, **train_config)