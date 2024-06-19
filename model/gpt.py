import jax
import jax.numpy as jnp
import optax
from chex import dataclass
from flax import linen as nn

from .nn import forward


class FeedForwardBlock(nn.Module):
    hidden_dim: int
    drop_rate: float

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim)(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    num_heads: int
    hidden_dim: int
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
        x = FeedForwardBlock(self.hidden_dim, self.drop_rate)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    vocab_size: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float
    decode: bool

    @nn.compact
    def __call__(self, x, pos, mask, training=True):
        x = nn.Embed(self.vocab_size, self.hidden_dim)(x)
        pos_emb = nn.Embed(self.seq_len, self.hidden_dim)(pos)

        x = x + pos_emb
        x = nn.LayerNorm()(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate, self.decode)(x, mask, training=training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.vocab_size)(x)

        return x


class GPT(nn.Module):
    config: dataclass
    decode: bool = False

    @nn.compact
    def __call__(self, x, training=True):
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
            self.config.hidden_dim,
            self.config.num_heads,
            self.config.num_layers,
            self.config.drop_rate,
            self.decode
        )(x, pos, mask, training=training)


def get_optimizer(lr, weight_decay, warmup_steps, total_steps):
    lr = optax.cosine_onecycle_schedule(total_steps, lr, warmup_steps / total_steps, div_factor=25, final_div_factor=1000)
    return optax.adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay)


def get_cache(model, batch_size):
    return model.init({'params': jax.random.PRNGKey(0)}, None, jnp.zeros((batch_size, model.seq_len), dtype=int))['cache']


def generate_fn(params, state, key, batch_size, cache, model):
    state['cache'] = cache

    generated = jnp.empty((batch_size, model.seq_len, model.vocab_size), dtype=float)
    next_token = jnp.zeros((batch_size, 0), dtype=int)

    for i in range(model.seq_len):
        key, model_key, cat_key = jax.random.split(key, 3)
        logits, state = forward(model, params, state, model_key, next_token, False)
        generated = generated.at[:, i].set(logits[:, 0])
        next_token = jax.random.categorical(model_key, logits[:, 0])

    return generated
