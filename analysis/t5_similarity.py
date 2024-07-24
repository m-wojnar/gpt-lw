from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from gpt_lw.data import sample_batch, get_dataset
from gpt_lw.model_utils import load_pretrained_model, forward, init_cache


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--name", type=str, default="llama_wiki_mini_short")
    args = args.parse_args()

    batch_size = 64
    key = jax.random.PRNGKey(42)

    model, variables = load_pretrained_model(f"runs/{args.name}")
    seq_len = model.config.seq_len
    inputs = jnp.empty((batch_size, seq_len), dtype=int)
    cache = init_cache(model, inputs)

    all_tokens, tokenizer = get_dataset("text_dataset/val_wikipedia.npy", dataset_type="text")
    gen_fn = jax.jit(lambda key, context: forward(model, variables | {'cache': cache}, key, context, method="context_gen")[0])
    decode_fn = lambda tokens: [tokenizer.decode(t) for t in tokens]
    t5_model = SentenceTransformer("sentence-transformers/sentence-t5-large")

    sample_fn = jax.jit(partial(sample_batch, all_tokens, batch_size, seq_len + 1))
    n_steps = 2000
    cosine_sim = 0.0

    for i in tqdm(range(n_steps)):
        key, batch_key, model_key = jax.random.split(key, 3)

        xt, _ = sample_fn(batch_key)
        x_gen = gen_fn(model_key, xt[:, :seq_len // 2])

        text_true, text_gen = decode_fn(xt[:, seq_len // 2:]), decode_fn(x_gen)
        true_emb, gen_emb = t5_model.encode(text_true), t5_model.encode(text_gen)
        cosine_sim += t5_model.similarity(true_emb, gen_emb).diag().mean().item()

    cosine_sim /= n_steps
    print(f"{args.name} T5 cosine similarity: {cosine_sim:.4f}")
