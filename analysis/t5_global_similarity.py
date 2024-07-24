from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from gpt_lw.data import get_dataset
from gpt_lw.model_utils import load_pretrained_model, forward, init_cache


EOT_TOKEN_NL = "<|endoftext|>"


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
    all_text = tokenizer.decode(all_tokens).split(EOT_TOKEN_NL)[1:]
    all_text = [text for text in all_text if len(text) >= seq_len]
    all_text = np.array(all_text)

    gen_fn = jax.jit(lambda key, context: forward(model, variables | {'cache': cache}, key, context, method="context_gen")[0])
    decode_fn = lambda tokens: [tokenizer.decode(t) for t in tokens]
    t5_model = SentenceTransformer("sentence-transformers/sentence-t5-base")

    cosine_sim = 0.0
    n_steps = 2000

    for _ in tqdm(range(n_steps)):
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

    cosine_sim /= n_steps
    print(f"{args.name} T5 cosine similarity: {cosine_sim:.4f}")
