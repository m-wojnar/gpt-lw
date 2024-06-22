import os
import yaml
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import optax
from chex import Array

from gpt_lw.data import Tokenizer, get_dataset, sample_batch
from gpt_lw.loss import get_weighted_loss
from gpt_lw.model_utils import get_optimizer, init, gradient_step, save_model
from gpt_lw.model_zdc import GPT, GPTConfig


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
        val_freq: int,
        **kwargs
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
    sample_fn = jax.jit(partial(sample_batch, dataset, batch_size, config.seq_len))

    # train loop
    for step in range(n_steps):
        train_key, batch_key = jax.random.split(train_key)
        xt, xtp1 = sample_fn(batch_key)

        variables, opt_state, loss = step_fn(variables, (train_key, xt, xtp1), opt_state)
        print("loss: ", loss)

        if step % val_freq == 0:
            val_key, batch_key = jax.random.split(val_key)
            xt, xtp1 = sample_fn(batch_key)

            val_loss, _ = eval_fn(variables, val_key, xt, xtp1)
            print("val_loss: ", val_loss)

        if step % save_freq == 0:
            save_model(variables, opt_state, f"checkpoints/{run_name}/step_{step}.pkl")

    # save final model
    save_model(variables, opt_state, f"checkpoints/{run_name}/final.pkl")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--gpt_config", type=str, default="configs/gpt/base.yaml")
    args.add_argument("--optimizer_config", type=str, default="configs/optimizer/base.yaml")
    args.add_argument("--train_config", type=str, default="configs/train/base.yaml")
    args.add_argument("--run_name", type=str, default="base")
    args = args.parse_args()

    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    dataset, tokenizer = get_dataset(train_config["dataset_path"])

    with open(args.gpt_config) as f:
        gpt_config = yaml.safe_load(f)
        gpt_config["vocab_size"] = tokenizer.vocab_size

    gpt_config = GPTConfig(**gpt_config)

    with open(args.optimizer_config) as f:
        optimizer_config = yaml.safe_load(f)
        optimizer_config["n_steps"] = train_config["n_steps"]

    optimizer = get_optimizer(**optimizer_config)

    train(args.run_name, gpt_config, dataset, tokenizer, optimizer, **train_config)