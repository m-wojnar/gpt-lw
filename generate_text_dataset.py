import os
import re
import glob
import random

import pandas as pd
import jax.numpy as jnp
from tqdm import tqdm

from gpt_lw.data import TextTokenizer

EOT_TOKEN_NL = "<|endoftext|>"

if __name__ == "__main__":
    wikipedia_dir = "text_dataset/wikipedia/"
    dataset_name = "wikipedia"
    shuffle = True
    val_pages_pct = 0.10

    parquet_files = glob.glob(os.path.join(wikipedia_dir, "*.parquet"))

    all_text = []
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        all_text += df["text"].tolist()

    if shuffle:
        random.Random(42).shuffle(all_text)

    n_pages = len(all_text)
    train_n_pages, val_n_pages = int(n_pages * (1 - val_pages_pct)), int(n_pages * val_pages_pct)

    all_pages = [re.sub(r'(?<!\s)\n\n', ' \n\n', page) for page in all_text]
    tokenizer = TextTokenizer()

    eot_token = f' {EOT_TOKEN_NL}'
    delim_enc = tokenizer.encode(eot_token)
    pages_enc = [tokenizer.encode(page + eot_token) for page in tqdm(all_pages)]

    train_pages_enc = [delim_enc] + pages_enc[:train_n_pages]
    val_pages_enc = [delim_enc] + pages_enc[train_n_pages:train_n_pages + val_n_pages]

    # concat all tokens from train_pages_enc and val_pages_enc
    train_tokens = jnp.concatenate(train_pages_enc)
    val_tokens = jnp.concatenate(val_pages_enc)

    # Print some stats (n_tokens, avg tokens per page, etc.)
    tokens = len(train_tokens) + len(val_tokens)
    print(f"Number of pages: {len(all_pages)}")
    print(f"Number of tokens: {tokens}")
    print(f"Average tokens per page: {tokens / len(all_pages)}")

    train_loc = f"text_dataset/train_{dataset_name}.npy"
    print(f"Writing train tokens to {train_loc}")
    jnp.save(train_loc, train_tokens)

    val_loc = f"text_dataset/val_{dataset_name}.npy"
    print(f"Writing val tokens to {val_loc}")
    jnp.save(val_loc, val_tokens)
