import os
import math
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
from generate_cfg_dataset import EOT_TOKEN_CFG
from generate_text_dataset import EOT_TOKEN_NL
from gpt_lw.data import Tokenizer, get_dataset, sample_batch
from gpt_lw.grad_utils import grad_norm, grad_norm_per_token
from gpt_lw.loss import get_weighted_loss
from gpt_lw.model_utils import get_optimizer, init, init_cache, gradient_step, save_train_state, load_train_state, forward
from gpt_lw.model import GPT, GPTConfig


def train(
        run_name: str,
        config: GPTConfig,
        train_dataset: Array,
        val_dataset: Array,
        tokenizer: Tokenizer,
        optimizer: optax.GradientTransformation,
        schedule: optax.Schedule,
        loss_weighting: str,
        batch_size: int,
        gn_batch_size: int,
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

    train_state = {"variables": None, "opt_state": None, "misc_metrics": [], "step": 0}
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        train_state = load_train_state(train_state, checkpoint_path)
        variables = train_state["variables"]
        opt_state = train_state["opt_state"]
        init_step = train_state["step"]
        misc_metrics = train_state["misc_metrics"]
        print(f"Resuming from step {init_step}")
        print(f"Metrics collected: {len(misc_metrics)}")
    else:
        train_state["variables"] = variables = init(model, init_key, inputs)
        train_state["opt_state"] = opt_state = optimizer.init(variables["params"])
        train_state["init_step"] = init_step = 0
        train_state["misc_metrics"] = misc_metrics = []

    if init_step == n_steps:
        print("Model already trained for n_steps!")
        return

    n_params = sum(x.size for x in jax.tree.leaves(variables['params']))
    print(f"Model has {n_params} parameters")

    # compiled functions
    def mean_loss_fn(fn):
        def _fn(*args):
            loss, aux = fn(*args)
            return loss.mean(), aux
        return _fn

    loss_fn = get_weighted_loss(model, loss_weighting, delim_token=tokenizer.encode(EOT_TOKEN_NL).item())
    eval_fn = get_weighted_loss(model, "unweighted")  # CCE/compression

    per_token_gn_fn = jax.jit(partial(grad_norm_per_token, loss_fn, gn_batch_size))
    gn_fn = jax.jit(partial(grad_norm, mean_loss_fn(loss_fn)))
    step_fn = jax.jit(partial(gradient_step, loss_fn=mean_loss_fn(loss_fn), optimizer=optimizer))
    per_token_loss_fn = jax.jit(loss_fn)
    per_token_cce_fn = jax.jit(eval_fn)
    loss_fn = jax.jit(mean_loss_fn(loss_fn))
    eval_fn = jax.jit(mean_loss_fn(eval_fn))
    train_sample_fn = jax.jit(partial(sample_batch, train_dataset, batch_size, config.seq_len + 1))
    val_sample_fn = jax.jit(partial(sample_batch, val_dataset, batch_size, config.seq_len + 1))
    gn_sample_fn = jax.jit(partial(sample_batch, train_dataset, gn_batch_size, config.seq_len + 1))
    gen_fn = jax.jit(lambda variables, key: forward(model, variables | {'cache': cache}, key, batch_size, method="gen")[0])

    # train loop
    for step in range(init_step, n_steps):
        t0_train = time.time()
        log_dict = {'step': step, 'tokens': step * batch_size * config.seq_len}
        step_key, batch_key, train_key = jax.random.split(train_key, 3)
        xt, xtp1 = train_sample_fn(batch_key)

        variables, opt_state, loss = step_fn(variables, (step_key, xt, xtp1), opt_state)

        train_time = time.time() - t0_train
        log_dict["perf/train_time"] = train_time
        log_dict["perf/train_tokens_p_s"] = batch_size * config.seq_len / train_time

        log_dict["train/loss"] = loss.item()
        log_dict["train/lr"] = schedule(opt_state[-1].count)

        if step % val_freq == 0:
            t0_val = time.time()
            val_loss, val_cce, val_context_cce, val_mean_token_gn, val_global_gn = 0.0, 0.0, 0.0, 0.0, 0.0

            token_gn_accum = jnp.zeros((gn_batch_size, config.seq_len))
            token_loss_accum = jnp.zeros((gn_batch_size, config.seq_len))
            token_cce_accum = jnp.zeros((gn_batch_size, config.seq_len))

            for i in range(n_val_steps):
                loss_key, eval_key, grad_key, val_batch_key, grad_batch_key, val_key = jax.random.split(val_key, 6)

                xt, xtp1 = val_sample_fn(val_batch_key)
                val_loss_t, _ = loss_fn(variables, loss_key, xt, xtp1)
                val_cce_t, _ = eval_fn(variables, eval_key, xt, xtp1)
                val_context_cce_t, _ = eval_fn(variables, eval_key, xt[:, config.seq_len // 2:], xtp1[:, config.seq_len // 2:])
                val_loss += val_loss_t.item()
                val_cce += val_cce_t.item()
                val_context_cce += val_context_cce_t.item()

                xt, xtp1 = gn_sample_fn(grad_batch_key)
                grad_norms = per_token_gn_fn(variables, grad_key, xt, xtp1)
                global_gn = gn_fn(variables, grad_key, xt, xtp1)
                token_loss, _ = per_token_loss_fn(variables, grad_key, xt, xtp1)
                token_cce, _ = per_token_cce_fn(variables, grad_key, xt, xtp1)

                token_loss_accum += token_loss
                token_gn_accum += grad_norms
                token_cce_accum += token_cce
                val_mean_token_gn += grad_norms.mean().item()
                val_global_gn += global_gn.item()

                misc_metrics.append((xt, xtp1, token_loss, token_cce, grad_norms, step))

            token_loss_accum /= n_val_steps
            token_gn_accum /= n_val_steps
            token_cce_accum /= n_val_steps

            # log log-spaced points
            points = [0] + [2**p for p in range(math.floor(math.log2(token_loss_accum.shape[1])) + 1)]
            for p in points:
                log_dict[f"val_loss(pos)/token_loss_{p}"] = token_loss_accum[:, p].mean().item()
                log_dict[f"val_cce(pos)/token_cce_{p}"] = token_cce_accum[:, p].mean().item()
                log_dict[f"val_grad(pos)/token_gn_{p}"] = token_gn_accum[:, p].mean().item()

            log_dict["val/loss"] = val_loss / n_val_steps
            log_dict["val/cce"] = val_cce / n_val_steps
            log_dict["val/context_cce"] = val_context_cce / n_val_steps
            log_dict["val/mean_token_gn"] = val_mean_token_gn / n_val_steps
            log_dict["val/global_gn"] = val_global_gn / n_val_steps

            val_time = time.time() - t0_val
            log_dict["perf/val_time"] = val_time

        if step % log_freq == 0:
            print(log_dict)
            if logging == "wandb":
                wandb.log(log_dict)

        train_state = {
            "variables": variables,
            "opt_state": opt_state,
            "misc_metrics": misc_metrics,
            "step": step,
        }
        if step % save_freq == 0 and step > 0:
            if save_intermediate:
                save_train_state(train_state, path=f"runs/{run_name}/checkpoints/step_{step}")
            save_train_state(train_state, path=f"runs/{run_name}/checkpoints/last")
    save_train_state(train_state, path=f"runs/{run_name}/checkpoints/last")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--gpt_config", type=str, default="configs/gpt/debug.yaml")
    args.add_argument("--optimizer_config", type=str, default="configs/optimizer/debug.yaml")
    args.add_argument("--train_config", type=str, default="configs/train/debug_nl.yaml")
    args.add_argument("--checkpoint_path", type=str, default=None) # manual override of auto last checkpoint
    args.add_argument("--run_name", type=str, default="debug")
    args.add_argument("--loss_weighting", type=str, default="unweighted")
    args = args.parse_args()

    if not os.path.exists(f"runs/{args.run_name}/checkpoints/last_variables.pkl"):  # run does not exist, parse input configs
        print(f"Starting new run in runs/{args.run_name}...")
        with open(args.train_config) as f:
            train_config = yaml.safe_load(f)

        EOT_TOKEN = EOT_TOKEN_NL
        train_dataset, tokenizer = get_dataset(train_config["train_dataset_path"], dataset_type="text")
        val_dataset, _ = get_dataset(train_config["val_dataset_path"], dataset_type="text")

        with open(args.gpt_config) as f:
            gpt_config_base = yaml.safe_load(f)
        gpt_config_base["eot_token"] = tokenizer.encode(EOT_TOKEN).item()
        gpt_config_base["vocab_size"] = tokenizer.vocab_size
        model_dtype = gpt_config_base.pop("dtype")

        gpt_config = GPTConfig(
            dtype=getattr(jnp, model_dtype, float),
            **gpt_config_base
        )
        gpt_config_base['dtype'] = model_dtype  # return so we can save

        with open(args.optimizer_config) as f:
            optimizer_config = yaml.safe_load(f)
            optimizer_config["n_steps"] = train_config["n_steps"]

        optimizer, schedule = get_optimizer(**optimizer_config)

        # create dir structure
        os.makedirs(f"runs/{args.run_name}/checkpoints", exist_ok=True)
        os.makedirs(f"runs/{args.run_name}/configs", exist_ok=True)
        os.makedirs(f"runs/{args.run_name}/analysis", exist_ok=True)

        # save configs
        with open(f"runs/{args.run_name}/configs/gpt.yaml", "w") as f:
            yaml.dump(gpt_config_base, f)
        with open(f"runs/{args.run_name}/configs/optimizer.yaml", "w") as f:
            yaml.dump(optimizer_config, f)
        with open(f"runs/{args.run_name}/configs/train.yaml", "w") as f:
            yaml.dump(train_config, f)

    else: # run does exist, read configs (ignores input configs)
        print(f"Resuming run in runs/{args.run_name}...")
        with open(f"runs/{args.run_name}/configs/gpt.yaml") as f:
            gpt_config = yaml.safe_load(f)
            model_dtype = gpt_config.pop("dtype")
            gpt_config = GPTConfig(
                dtype=getattr(jnp, model_dtype, float),
                **gpt_config
            )
        with open(f"runs/{args.run_name}/configs/optimizer.yaml") as f:
            optimizer_config = yaml.safe_load(f)
        with open(f"runs/{args.run_name}/configs/train.yaml") as f:
            train_config = yaml.safe_load(f)

        optimizer, schedule = get_optimizer(**optimizer_config)
        train_dataset, tokenizer = get_dataset(train_config["train_dataset_path"], dataset_type="text")
        val_dataset, _ = get_dataset(train_config["val_dataset_path"], dataset_type="text")

        if args.checkpoint_path is None:  # manual path has top priority
            args.checkpoint_path = f"runs/{args.run_name}/checkpoints/last"

    with jax.disable_jit(train_config['debug']):
        train(
            run_name=args.run_name,
            config=gpt_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            optimizer=optimizer,
            schedule=schedule,
            loss_weighting=args.loss_weighting,
            checkpoint_path=args.checkpoint_path,
            **train_config
        )
