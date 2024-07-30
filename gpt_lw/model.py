from dataclasses import dataclass
from functools import partial

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
    eot_token: int
    dtype: jnp.dtype
    embed_init: nn.initializers.Initializer = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()


class FeedForwardBlock(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, training=True):
        dense = partial(nn.Dense, dtype=self.config.dtype, use_bias=False, kernel_init=self.config.kernel_init)
        out_dim = x.shape[-1]

        x = dense(self.config.ff_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.config.drop_rate)(x, deterministic=not training)
        x = dense(out_dim)(x)
        x = nn.Dropout(self.config.drop_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm(dtype=self.config.dtype, use_bias=False)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=x.shape[-1],
            decode=not training,
            dtype=self.config.dtype,
            kernel_init=self.config.kernel_init,
            use_bias=False
        )(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm(dtype=self.config.dtype, use_bias=False)(x)
        x = FeedForwardBlock(self.config)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    config: GPTConfig

    def setup(self) -> None:
        self.token_emb = nn.Embed(self.config.vocab_size, self.config.embed_dim, embedding_init=self.config.embed_init)
        self.pos_emb = nn.Embed(self.config.seq_len, self.config.embed_dim, embedding_init=self.config.embed_init)
        self.t_blocks = [TransformerBlock(self.config) for _ in range(self.config.num_layers)]
        self.out_ln = nn.LayerNorm(dtype=self.config.dtype, use_bias=False)

    def __call__(self, x, pos, mask, training=True):
        x = self.token_emb(x)
        x = x + self.pos_emb(pos)

        for block in self.t_blocks:
            x = block(x, mask, training=training)

        x = self.out_ln(x)
        x = self.token_emb.attend(x.astype(jnp.float32))

        return x


class GPT(nn.Module):
    config: dataclass

    @nn.compact
    def __call__(self, x, training=True):
        if not training:
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
            mask = nn.make_causal_mask(x, dtype=self.config.dtype)

        return Transformer(self.config)(x, pos, mask, training=training)

    def gen(self, batch_size):
        def scan_fn(gpt, carry):
            prev_token, key = carry
            key, cat_key = jax.random.split(key)

            logits = gpt(prev_token, training=False)
            next_token = jax.random.categorical(cat_key, logits)
            return (next_token, key), next_token

        scan = nn.scan(scan_fn, variable_broadcast='params', variable_carry='cache', out_axes=1, length=self.config.seq_len)
        first_token = jnp.ones((batch_size, 1), dtype=int) * self.config.eot_token
        _, generated = scan(self, (first_token, self.make_rng('gpt')))
        return generated.squeeze()

    def context_gen(self, context):
        def scan_fn(gpt, carry):
            prev_token, idx, key = carry
            prev_token = jax.lax.select(idx < context.shape[1], context[:, idx][:, None], prev_token)
            key, cat_key = jax.random.split(key)

            logits = gpt(prev_token, training=False)
            next_token = jax.random.categorical(cat_key, logits).astype(jnp.uint16)
            return (next_token, idx + 1, key), next_token

        scan = nn.scan(scan_fn, variable_broadcast='params', variable_carry='cache', out_axes=1, length=self.config.seq_len)
        empty_token = jnp.empty((context.shape[0], 1), dtype=jnp.uint16)
        _, generated = scan(self, (empty_token, 0, self.make_rng('gpt')))
        return generated[:, context.shape[1]:].squeeze()
