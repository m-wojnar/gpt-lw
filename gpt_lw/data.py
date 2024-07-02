import jax
import jax.numpy as jnp
import tiktoken
import tokenmonster
from chex import Array, PRNGKey


class Tokenizer:
    def encode(self, text: str) -> Array:
        raise NotImplementedError

    def decode(self, tokens: Array) -> str:
        raise NotImplementedError


class SimpleTokenizer(Tokenizer):
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


class TextTokenizer(Tokenizer):
    def __init__(self):
        self.enc = tokenmonster.load("english-2048-clean-v1")
        self.enc.add_special_token("<|endoftext|>")
        self.vocab_size = self.enc.vocab_size
        # self.enc = tiktoken.get_encoding("cl100k_base")
        # self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> Array:
        return jnp.array(self.enc.tokenize(text))
        # return jnp.array(self.enc.encode(text, allowed_special={'<|endoftext|>'}))

    def decode(self, tokens: Array) -> str:
        return self.enc.decode(tokens.tolist())


def get_dataset(path: str, dataset_type="cfg") -> tuple[Array, Tokenizer]:
    if dataset_type == "cfg":
        with open(path, "r") as f:
            data = f.read()
        tokens = sorted(list(set(data)))
        tokenizer = SimpleTokenizer(tokens)
        data_e = tokenizer.encode(data)
    else:
        # load as jnp array
        data_e = jnp.load(path)
        tokenizer = TextTokenizer()

    return data_e, tokenizer


def sample_batch(input_tensor: Array, batch_size: int, seq_len: int, key: PRNGKey) -> tuple[Array, Array]:
    N = input_tensor.shape[0]
    indices = jax.random.randint(key, (batch_size,), minval=0, maxval=N - seq_len + 1)
    batch = jax.vmap(lambda i: jax.lax.dynamic_slice(input_tensor, (i,), (seq_len,)))(indices)
    return batch[:, :-1], batch[:, 1:]