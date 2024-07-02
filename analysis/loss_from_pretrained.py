import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM

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


def torch_to_numpy(*tensors):
    return [t.detach().cpu().numpy() for t in tensors]


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len, device):
        self.tokens = tokens
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        return self.tokens[idx:idx + self.seq_len].to(self.device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)

    text = EOT_TOKEN_NL.join(load_text_data())
    all_tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    text_dataset = TextDataset(all_tokens, seq_len=512 + 1, device=device)
    data_loader = DataLoader(text_dataset, batch_size=64)

    history = []

    with torch.no_grad():
        for tokens in data_loader:
            xt, xtp1 = tokens[:, :-1], tokens[:, 1:]
            output = model.forward(xt, use_cache=True).logits.permute(0, 2, 1)
            loss = F.cross_entropy(output, xtp1, reduction="none")
            history.append(torch_to_numpy(xt, xtp1, loss))

    xt_history, xtp1_history, loss_history = zip(*history)
    xt_history, xtp1_history, loss_history = np.concatenate(xt_history), np.concatenate(xtp1_history), np.concatenate(loss_history)
    np.savez("history.npz", xt=xt_history, xtp1=xtp1_history, loss=loss_history)
