import os
import time
from argparse import ArgumentParser
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import optax
import wandb
import yaml
from chex import Array

from cfg_dataset.cfg import CFG
from generate_dataset import DELIM_TOKEN
from gpt_lw.data import Tokenizer, get_dataset, sample_batch
from gpt_lw.loss import get_weighted_loss
from gpt_lw.model_utils import get_optimizer, init, init_cache, gradient_step, save_model, load_model, forward
from gpt_lw.model_zdc import GPT, GPTConfig


def train(
        run_name: str,
        config: GPTConfig,
        train_dataset: Array,
        val_dataset: Array,
        cfg: CFG,
        tokenizer: Tokenizer,
        optimizer: optax.GradientTransformation,
        schedule: optax.Schedule,
        loss_weighting: str,
        batch_size: int,
        n_steps: int,
        seed: int,
        save_freq: int,
        save_intermediate: bool,
        val_freq: int,
        n_val_steps: int,
        log_freq: int,
        logging: Literal["wandb", "stdout"],
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

    if logging == "wandb":
        wandb.init(project="gpt-lw", entity="gpt-lw", dir=f"runs/{run_name}", name=run_name)

    # gen random keys
    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    # check device
    device = jax.devices()[0].platform
    print(f"Device: {device}")

    # init model
    model = GPT(config)
    inputs = jnp.empty((batch_size, config.seq_len), dtype=int)
    cache = init_cache(model, inputs)

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        variables, opt_state, init_step = load_model(checkpoint_path)
    else:
        variables = init(model, init_key, inputs)
        opt_state = optimizer.init(variables["params"])
        init_step = 0

    if init_step == n_steps:
        print("Model already trained for n_steps!")
        return

    n_params = sum(x.size for x in jax.tree.leaves(variables['params']))
    print(f"Model has {n_params} parameters")

    # compiled functions
    loss_fn = get_weighted_loss(model, loss_weighting)
    eval_fn = get_weighted_loss(model, "unweighted")  # CCE/compression

    step_fn = jax.jit(partial(gradient_step, loss_fn=loss_fn, optimizer=optimizer))
    loss_fn = jax.jit(loss_fn)
    eval_fn = jax.jit(eval_fn)
    train_sample_fn = jax.jit(partial(sample_batch, train_dataset, batch_size, config.seq_len))
    val_sample_fn = jax.jit(partial(sample_batch, val_dataset, batch_size, config.seq_len))
    gen_fn = jax.jit(lambda variables, key: forward(model, variables | {'cache': cache}, key, method="gen")[0])

    # train loop
    for step in range(init_step, n_steps):
        t0_train = time.time()
        log_dict = {'step': step, 'tokens': step * batch_size * config.seq_len}
        train_key, batch_key = jax.random.split(train_key)
        xt, xtp1 = train_sample_fn(batch_key)

        variables, opt_state, loss = step_fn(variables, (train_key, xt, xtp1), opt_state)

        train_time = time.time() - t0_train
        log_dict["train/time"] = train_time

        log_dict["train/loss"] = loss.item()
        log_dict["train/lr"] = schedule(opt_state[-1].count)

        t0_val = time.time()
        if step % val_freq == 0:
            val_loss, val_cce = 0.0, 0.0

            for i in range(n_val_steps):
                val_key, batch_key = jax.random.split(val_key)
                xt, xtp1 = val_sample_fn(batch_key)
                val_loss_t, _ = loss_fn(variables, val_key, xt, xtp1)
                val_cce_t, _ = eval_fn(variables, val_key, xt, xtp1)
                val_loss += val_loss_t.item()
                val_cce += val_cce_t.item()

            log_dict["val/loss"] = val_loss / n_val_steps
            log_dict["val/cce"] = val_cce / n_val_steps

            # CFG accuracy eval:
            gen_tokens = gen_fn(variables, val_key)
            tot_cfg_samples = sum((tokenizer.decode(t).split(',')[1:-1] for t in gen_tokens), start=[])
            print(len(tot_cfg_samples))

            cfg_acc = sum([cfg.verify(s) for s in tot_cfg_samples]) / len(tot_cfg_samples)
            log_dict["val/cfg_acc"] = cfg_acc
        val_time = time.time() - t0_val
        log_dict["val/time"] = val_time

        if step % log_freq == 0:
            print(log_dict)
            if logging == "wandb":
                wandb.log(log_dict)

        if step % save_freq == 0 and step > 0:
            if save_intermediate:
                save_model(variables, opt_state, step, f"runs/{run_name}/checkpoints/step_{step}.pkl")
            save_model(variables, opt_state, step, f"runs/{run_name}/checkpoints/last.pkl")  # last checkpoint

    save_model(variables, opt_state, n_steps, f"runs/{run_name}/checkpoints/last.pkl")  # final checkpoint


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--gpt_config", type=str, default="configs/gpt/debug.yaml")
    args.add_argument("--optimizer_config", type=str, default="configs/optimizer/debug.yaml")
    args.add_argument("--train_config", type=str, default="configs/train/debug.yaml")
    args.add_argument("--run_name", type=str, default="debug")
    args.add_argument("--loss_weighting", type=str, default="unweighted")
    args = args.parse_args()

    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    cfg = CFG(rules_file=f"configs/cfg/{train_config['cfg_name']}.cfg")
    train_dataset, tokenizer = get_dataset(train_config["train_dataset_path"])
    val_dataset, _ = get_dataset(train_config["val_dataset_path"])

    with open(args.gpt_config) as f:
        gpt_config = yaml.safe_load(f)
        gpt_config["gen_batch_size"] = train_config["gen_batch_size"]
        gpt_config["delim_token"] = tokenizer.encode(DELIM_TOKEN).item()
        gpt_config["vocab_size"] = tokenizer.vocab_size

    gpt_config = GPTConfig(**gpt_config)

    with open(args.optimizer_config) as f:
        optimizer_config = yaml.safe_load(f)
        optimizer_config["n_steps"] = train_config["n_steps"]

    optimizer, schedule = get_optimizer(**optimizer_config)

    train(args.run_name, gpt_config, train_dataset, val_dataset, cfg, tokenizer, optimizer, schedule, args.loss_weighting, **train_config)