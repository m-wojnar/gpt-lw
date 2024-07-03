from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import yaml
from chex import Array
from tqdm import trange

from generate_text_dataset import EOT_TOKEN_NL
from gpt_lw.data import sample_batch, get_dataset, Tokenizer
from gpt_lw.loss import get_weighted_loss, compute_relative_positions
from gpt_lw.model import GPTConfig, GPT
from gpt_lw.model_utils import load_variables


def run(
        config: GPTConfig,
        train_dataset: Array,
        tokenizer: Tokenizer,
        checkpoint_path: str,
        batch_size: int,
        n_steps: int,
        seed: int,
        **kwargs
    ):

    model = GPT(config)
    variables, *_ = load_variables(checkpoint_path)

    loss_fn = jax.jit(get_weighted_loss(model, "unweighted"))
    sample_fn = jax.jit(partial(sample_batch, train_dataset, batch_size, config.seq_len + 1))
    relpos_fn = jax.jit(compute_relative_positions)

    key = jax.random.PRNGKey(seed)
    losses = jnp.zeros((tokenizer.vocab_size, config.seq_len))
    counter = jnp.zeros((tokenizer.vocab_size, 1), dtype=jnp.int32)

    for _ in trange(n_steps):
        key, model_key, batch_key = jax.random.split(key, 3)
        xt, xtp1 = sample_fn(batch_key)
        loss, _ = loss_fn(variables, model_key, xt, xtp1)

        for token in jnp.unique(xt):
            mask = (xt == token).any(axis=1)
            Xs = xt[mask]
            Ls = loss[mask]
            RPs = relpos_fn(Xs, token)

            losses = losses.at[token, RPs].add(Ls)
            counter = counter.at[token].add(mask.sum())

    losses /= counter
    jnp.save("losses.npy", losses)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--n_steps", type=int, default=100)
    args.add_argument("--checkpoint_path", type=str, default="../runs/debug/checkpoints/last.pkl")
    args.add_argument("--gpt_config", type=str, default="../configs/gpt/debug.yaml")
    args.add_argument("--train_config", type=str, default="../configs/train/debug_nl.yaml")
    args = args.parse_args()

    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
        train_config["n_steps"] = args.n_steps

    train_dataset, tokenizer = get_dataset(f'../{train_config["train_dataset_path"]}', dataset_type="text")

    with open(args.gpt_config) as f:
        gpt_config = yaml.safe_load(f)
        gpt_config["gen_batch_size"] = train_config["gen_batch_size"]
        gpt_config["eot_token"] = tokenizer.encode(EOT_TOKEN_NL).item()
        gpt_config["vocab_size"] = tokenizer.vocab_size
        gpt_config["dtype"] = getattr(jnp, gpt_config["dtype"], float)

    gpt_config = GPTConfig(**gpt_config)

    run(gpt_config, train_dataset, tokenizer, args.checkpoint_path, **train_config)
