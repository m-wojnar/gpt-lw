from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from sentence_transformers import SentenceTransformer

from gpt_lw.data import sample_batch, get_dataset
from gpt_lw.model_utils import load_pretrained_model, forward, init_cache


EOT_TOKEN_NL = "<|endoftext|>"


def t5_global_similarity(name):
    batch_size = 64
    key = jax.random.PRNGKey(42)

    model, variables = load_pretrained_model(f"runs/{name}")
    seq_len = model.config.seq_len
    inputs = jnp.empty((batch_size, seq_len), dtype=int)
    cache = init_cache(model, inputs)

    all_tokens, tokenizer = get_dataset("text_dataset/val_wikipedia.npy", dataset_type="text")
    delim_token = tokenizer.encode(EOT_TOKEN_NL)
    all_tokens = jnp.split(all_tokens, jnp.where(all_tokens == delim_token)[0] + 1)[1:]
    all_tokens = [t for t in all_tokens if len(t) >= seq_len]
    all_text = np.array([tokenizer.decode(t) for t in all_tokens])

    gen_fn = jax.jit(lambda key, context: forward(model, variables | {'cache': cache}, key, context, method="context_gen")[0])
    decode_fn = lambda tokens: [tokenizer.decode(t) for t in tokens]
    t5_model = SentenceTransformer("sentence-transformers/sentence-t5-base")

    cosine_sim = 0.0
    n_steps = 2000

    for _ in trange(n_steps):
        key, batch_key, seq_key, model_key = jax.random.split(key, 4)
        batch_idx = jax.random.randint(batch_key, (batch_size,), 0, len(all_text))

        texts, tokens = all_text[batch_idx], []

        for text in texts:
            seq_key, subkey = jax.random.split(seq_key)
            encoded = tokenizer.encode(text)
            idx = jax.random.randint(subkey, (1,), 0, len(encoded) - seq_len).item()
            tokens.append(encoded[idx:idx + seq_len // 2])

        context = jnp.asarray(tokens)
        x_gen = gen_fn(model_key, context)

        text_gen = decode_fn(x_gen)
        true_emb, gen_emb = t5_model.encode(texts), t5_model.encode(text_gen)
        cosine_sim += t5_model.similarity(true_emb, gen_emb).diag().mean().item()

    return cosine_sim / n_steps


def t5_local_similarity(name):
    batch_size = 64
    key = jax.random.PRNGKey(42)

    model, variables = load_pretrained_model(f"runs/{name}")
    seq_len = model.config.seq_len
    inputs = jnp.empty((batch_size, seq_len), dtype=int)
    cache = init_cache(model, inputs)

    all_tokens, tokenizer = get_dataset("text_dataset/val_wikipedia.npy", dataset_type="text")
    gen_fn = jax.jit(lambda key, context: forward(model, variables | {'cache': cache}, key, context, method="context_gen")[0])
    decode_fn = lambda tokens: [tokenizer.decode(t) for t in tokens]
    t5_model = SentenceTransformer("sentence-transformers/sentence-t5-base")

    sample_fn = jax.jit(partial(sample_batch, all_tokens, batch_size, seq_len + 1))
    n_steps = 2000
    cosine_sim = 0.0

    for _ in trange(n_steps):
        key, batch_key, model_key = jax.random.split(key, 3)

        xt, _ = sample_fn(batch_key)
        x_gen = gen_fn(model_key, xt[:, :seq_len // 2])

        text_true, text_gen = decode_fn(xt[:, seq_len // 2:]), decode_fn(x_gen)
        true_emb, gen_emb = t5_model.encode(text_true), t5_model.encode(text_gen)
        cosine_sim += t5_model.similarity(true_emb, gen_emb).diag().mean().item()

    return cosine_sim / n_steps


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--name", type=str, default="llama_wiki_mini_short")
    args = args.parse_args()

    cosine_sim = t5_global_similarity(args.name)
    print(f"{args.name} T5 global cosine similarity: {cosine_sim:.4f}")

    cosine_sim = t5_local_similarity(args.name)
    print(f"{args.name} T5 local cosine similarity: {cosine_sim:.4f}")
