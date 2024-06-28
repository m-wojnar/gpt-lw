import jax
import jax.numpy as jnp
from chex import Array, PRNGKey


class Tokenizer:
    def __init__(self, tokens: list[str]):
        tokens = sorted(tokens) # make sure order is deterministic
        self.vocab_size = len(tokens)

        self.ctoi = dict([(c, i) for i, c in enumerate(tokens)])
        self.itoc = dict([(i, c) for i, c in enumerate(tokens)])

    def encode(self, text: str) -> Array:
        t_enc = [self.ctoi[c] for c in text]
        return jnp.array(t_enc)

    def decode(self, tokens: Array) -> str:
        tokens = tokens.tolist()
        chars = [self.itoc[t] for t in tokens]
        return "".join(chars)


def get_dataset(txt_path: str) -> tuple[Array, Tokenizer]:
    with open(txt_path, "r") as f:
        data = f.read()
    tokens = sorted(list(set(data)))
    tokenizer = Tokenizer(tokens)
    data_e = tokenizer.encode(data)
    return data_e, tokenizer


def sample_batch(input_tensor: Array, weights: Array, batch_size: int, seq_len: int, key: PRNGKey) -> tuple[Array, Array, Array]:
    N = input_tensor.shape[0]
    indices = jax.random.randint(key, (batch_size,), minval=0, maxval=N - seq_len + 1)
    batch = jax.vmap(lambda i: jax.lax.dynamic_slice(input_tensor, (i,), (seq_len,)))(indices)
    weights = jax.vmap(lambda i: jax.lax.dynamic_slice(weights, (i,), (seq_len,)))(indices)
    return batch[:, :-1], batch[:, 1:], weights[:, :-1]