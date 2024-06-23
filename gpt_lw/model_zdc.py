from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int
    embed_dim: int
    ff_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float


class FeedForwardBlock(nn.Module):
    ff_dim: int
    drop_rate: float

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.ff_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim)(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    ff_dim: int
    drop_rate: float
    decode: bool = False

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=x.shape[-1], decode=self.decode)(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForwardBlock(self.ff_dim, self.drop_rate)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    vocab_size: int
    seq_len: int
    embed_dim: int
    ff_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float
    decode: bool

    @nn.compact
    def __call__(self, x, pos, mask, training=True):
        x = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pos_emb = nn.Embed(self.seq_len, self.embed_dim)(pos)

        x = x + pos_emb
        x = nn.LayerNorm()(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, self.ff_dim, self.drop_rate, self.decode)(x, mask, training=training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.vocab_size)(x)

        return x


class GPT(nn.Module):
    config: dataclass
    decode: bool = False
    gen_batch_size: int = 1
    delim_token: int = 0

    @nn.compact
    def __call__(self, x):
        if self.decode:
            is_initialized = self.has_variable('cache', 'cache_index')
            cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=int))

            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
            else:
                i = jnp.array(0, dtype=int)

            pos = i
            mask = None
        else:
            pos = jnp.arange(x.shape[1])
            mask = nn.make_causal_mask(x)

        return Transformer(
            self.config.vocab_size,
            self.config.seq_len,
            self.config.embed_dim,
            self.config.ff_dim,
            self.config.num_heads,
            self.config.num_layers,
            self.config.drop_rate,
            self.decode
        )(x, pos, mask, training=not self.decode)

    # NOTE: not adding any fancy logit wrappers (top_k, top_p, etc.) here since
    # vocab size is probably too small for it to be relevant
    def gen(self):
        def scan_fn(gpt, carry):
            prev_token, key = carry
            key, cat_key = jax.random.split(key)

            logits = gpt(prev_token)
            next_token = jax.random.categorical(cat_key, logits)
            return (next_token, key), next_token

        scan = nn.scan(scan_fn, variable_broadcast='params', variable_carry='cache', out_axes=1, length=self.config.seq_len)
        first_token = jnp.ones((self.gen_batch_size, 1), dtype=int) * self.delim_token
        _, generated = scan(self, (first_token, self.make_rng('gpt')))
        return generated.squeeze()