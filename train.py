import os
import yaml
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import optax
from chex import Array

from cfg_dataset.cfg import CFG
from gpt_lw.data import Tokenizer, get_dataset, sample_batch
from gpt_lw.loss import get_weighted_loss
from gpt_lw.model_utils import get_optimizer, init, gradient_step, save_model, load_model, sample_model
from gpt_lw.model_zdc import GPT, GPTConfig


def train(
        run_name: str,
        config: GPTConfig,
        dataset: Array,
        cfg: CFG,
        tokenizer: Tokenizer,
        optimizer: optax.GradientTransformation,
        batch_size: int,
        n_steps: int,
        loss_weighting: str,
        seed: int,
        save_freq: int,
        val_freq: int,
        log_freq: int,
        checkpoint_path: str = None,
        **kwargs
    ):

    # TODO: expand run dir
    # - auto run name if one not passed
    # - save config in dir
    # - logs (.out/.err, loss/acc curves) subdir
    if not os.path.exists(f"runs/{run_name}"):
        os.makedirs(f"runs/{run_name}/checkpoints")
    else: # autoresume
        if checkpoint_path is None:  # manual path has top priority
            last_path = f"runs/{run_name}/checkpoints/last.pkl"
            if os.path.exists(last_path):
                checkpoint_path = last_path

    # gen random keys
    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    # check device
    device = jax.devices()[0].platform
    print("Device: ", device)

    # init model
    model = GPT(config)
    inputs = jnp.empty((batch_size, config.seq_len), dtype=int)

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        ckpt = load_model(checkpoint_path)
        variables, opt_state, init_step = ckpt['variables'], ckpt['opt_state'], ckpt['step']
    else:
        variables = init(model, init_key, inputs)
        opt_state = optimizer.init(variables["params"])
        init_step = 0

    n_params = sum(x.size for x in jax.tree.leaves(variables['params']))
    print(f"Model has {n_params} parameters")


    # gradient step and eval functions
    loss_fn = get_weighted_loss(model, loss_weighting)
    eval_fn = get_weighted_loss(model, "unweighted")  # CCE/compression
    step_fn = jax.jit(partial(gradient_step, loss_fn=loss_fn, optimizer=optimizer))
    eval_fn = jax.jit(eval_fn)
    sample_fn = jax.jit(partial(sample_batch, dataset, batch_size, config.seq_len))

    # train loop
    for step in range(init_step, n_steps):
        log_dict = {}
        train_key, batch_key = jax.random.split(train_key)
        xt, xtp1 = sample_fn(batch_key)

        variables, opt_state, loss = step_fn(variables, (train_key, xt, xtp1), opt_state)
        log_dict["loss"] = loss

        if step % val_freq == 0:
            val_loss = 0.0
            n_val_steps = 10
            for i in range(n_val_steps):  # TODO: make this a hyperparam (n_val_steps)
                val_key, batch_key = jax.random.split(val_key)
                xt, xtp1 = sample_fn(batch_key)
                val_loss_t, _ = eval_fn(variables, val_key, xt, xtp1)
                val_loss += val_loss_t
            log_dict["val_loss"] = val_loss / n_val_steps

            # TODO: fix CFG classes then uncomment
            # CFG accuracy eval:
            # TODO: get delim token from cfg
            # gen_tokens = sample_model(model, variables, val_key, batch_size, 2 * config.seq_len, 0)
            # gen_tokens = sample_model(model, variables, val_key, 1, config.seq_len, 0)

            # tot_cfg_samples = []
            # for i in range(gen_tokens.shape[0]):
            #     sample = tokenizer.decode(gen_tokens[i])
            #     print(sample)
            #     cfg_samples = sample.split(',')[1:-1]
            #     tot_cfg_samples += cfg_samples

            # cfg_acc = sum([cfg.verify(s) for s in tot_cfg_samples]) / len(tot_cfg_samples)
            # log_dict["cfg_acc"] = cfg_acc

        if step % log_freq == 0:
            print(f"Step {step}: {log_dict}")
        if step % save_freq == 0 and step > 0:
            if save_freq > 0:  # intermediate checkpoints
                save_model(variables, opt_state, step, f"runs/{run_name}/checkpoints/step_{step}.pkl")
            save_model(variables, opt_state, step, f"runs/{run_name}/checkpoints/last.pkl")  # last checkpoint
    save_model(variables, opt_state, step, f"runs/{run_name}/checkpoints/last.pkl")  # final checkpoint

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--gpt_config", type=str, default="configs/gpt/base.yaml")
    args.add_argument("--optimizer_config", type=str, default="configs/optimizer/base.yaml")
    args.add_argument("--train_config", type=str, default="configs/train/base.yaml")
    args.add_argument("--run_name", type=str, default="base")
    args = args.parse_args()

    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    cfg = CFG(rules_file=f"configs/cfg/{train_config['cfg_name']}.cfg")
    dataset, tokenizer = get_dataset(train_config["dataset_path"])

    with open(args.gpt_config) as f:
        gpt_config = yaml.safe_load(f)
        gpt_config["vocab_size"] = tokenizer.vocab_size

    gpt_config = GPTConfig(**gpt_config)

    with open(args.optimizer_config) as f:
        optimizer_config = yaml.safe_load(f)
        optimizer_config["n_steps"] = train_config["n_steps"]

    optimizer = get_optimizer(**optimizer_config)

    train(args.run_name, gpt_config, dataset, cfg, tokenizer, optimizer, **train_config)