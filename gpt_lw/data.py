import jax
import jax.numpy as jnp


class Tokenizer:
    def __init__(self, tokens: list[str]):
        tokens = sorted(tokens) # make sure order is deterministic
        self.vocab_size = len(tokens)

        self.ctoi = dict([(c, i) for i, c in enumerate(tokens)])
        self.itoc = dict([(i, c) for i, c in enumerate(tokens)])

    def encode(self, text: str) -> jnp.array:
        t_enc = [self.ctoi[c] for c in text]
        return jnp.array(t_enc)

    def decode(self, tokens: jnp.array) -> str:
        tokens = tokens.tolist()
        chars = [self.itoc[t] for t in tokens]
        return "".join(chars)


def get_dataset(txt_path: str) -> jnp.array:
    with open(txt_path, "r") as f:
        data = f.read()
    tokens = sorted(list(set(data)))
    tokenizer = Tokenizer(tokens)
    data_e = tokenizer.encode(data)
    return data_e, tokenizer


def sample_batch(input_tensor, batch_size, seq_len, key):
    N = input_tensor.shape[0]
    indices = jax.random.randint(key, (batch_size,), minval=0, maxval=N - seq_len + 1)
    batch = jnp.array([input_tensor[i:i + seq_len] for i in indices])
    return batch