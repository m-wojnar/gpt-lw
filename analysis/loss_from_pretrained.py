import os
import glob
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
import optax
from tqdm import tqdm
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

from gpt_lw.data import sample_batch, get_dataset
from gpt_lw.model_utils import load_pretrained_model, forward

from gpt_lw.grad_utils import grad_norm_per_token
from gpt_lw.loss import get_weighted_loss

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
    model_type = "gpt-lw"
    batch_size, seq_len = 64, 128
    gn_batch_size = 16
    compute_grad_norm = False

    for name in [
        'llama_wiki_mini_short_abs', 'llama_wiki_mini_short_abs_4', 'llama_wiki_mini_short_abs_8',
        'llama_wiki_mini_short_rel', 'llama_wiki_mini_short_rel_4', 'llama_wiki_mini_short_rel_8',
        'llama_wiki_mini_short_rel_nn', 'llama_wiki_mini_short_abs_random', 'llama_wiki_mini_short',
    ]:
        key = jax.random.PRNGKey(42)

        if model_type == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = FlaxAutoModelForCausalLM.from_pretrained("gpt2")
            text = EOT_TOKEN_NL.join(load_text_data())
            all_tokens = tokenizer(text, return_tensors="jax").input_ids[0]

            model_fn = jax.jit(lambda x, k: model(x).logits)
        elif model_type == "gpt-lw":
            all_tokens, tokenizer = get_dataset("text_dataset/train_wikipedia.npy", dataset_type="text")
            model, variables = load_pretrained_model(f"runs/{name}")

            loss_fn = get_weighted_loss(model, "unweighted")
            grad_norm_fn = jax.jit(partial(grad_norm_per_token, loss_fn, gn_batch_size))
            model_fn = jax.jit(lambda x, k: forward(model, variables, k, x)[0])

        sample_fn = jax.jit(partial(sample_batch, all_tokens, batch_size, seq_len + 1))

        history = []
        n_steps = 200

        for i in tqdm(range(n_steps)):
            key, batch_key, model_key = jax.random.split(key, 3)
            xt, xtp1 = sample_fn(batch_key)

            if compute_grad_norm:
                assert model_type == "gpt-lw"
                loss, grads = grad_norm_fn(variables, model_key, xt, xtp1)
                history.append((xt, xtp1, loss, grads))
            else:
                logits = model_fn(xt, model_key)
                loss = jax.jit(optax.losses.softmax_cross_entropy_with_integer_labels)(logits, xtp1)
                history.append((xt, xtp1, loss, jnp.zeros_like(loss)))


        xt_history, xtp1_history, loss_history, grad_history = zip(*history)
        xt_history, xtp1_history, loss_history, grad_history = jnp.concatenate(xt_history), jnp.concatenate(xtp1_history), jnp.concatenate(loss_history), jnp.concatenate(grad_history)
        jnp.savez(f"{name}_history.npz", xt=xt_history, xtp1=xtp1_history, loss=loss_history, grad=grad_history)