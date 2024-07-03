import os
import glob
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
import optax
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from gpt_lw.data import sample_batch


EOT_TOKEN_NL = "<|endoftext|>"


def load_text_data(dir="../text_dataset/wikipedia/", n_pages=1000):
    parquet_files = glob.glob(os.path.join(dir, "**", "*.parquet"), recursive=True)
    all_text = []

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        all_text += df["text"].tolist()

        if len(all_text) >= n_pages:
            break

    return all_text[:n_pages]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = FlaxAutoModelForCausalLM.from_pretrained("gpt2")

    text = EOT_TOKEN_NL.join(load_text_data())
    batch_size, seq_len = 64, 1024
    key = jax.random.PRNGKey(42)

    all_tokens = tokenizer(text, return_tensors="jax").input_ids[0]
    sample_fn = jax.jit(partial(sample_batch, all_tokens, batch_size, seq_len + 1))
    model_fn = jax.jit(lambda x: model(x).logits)

    history = []
    n_steps = 1000

    for i in tqdm(range(n_steps)):
        key, batch_key = jax.random.split(key)
        xt, xtp1 = sample_fn(batch_key)
        logits = model_fn(xt)
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits, xtp1)
        history.append((xt, xtp1, loss))

    xt_history, xtp1_history, loss_history = zip(*history)
    xt_history, xtp1_history, loss_history = jnp.concatenate(xt_history), jnp.concatenate(xtp1_history), jnp.concatenate(loss_history)
    jnp.savez("history.npz", xt=xt_history, xtp1=xtp1_history, loss=loss_history)
