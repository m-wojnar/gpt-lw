import jax
import jax.numpy as jnp
import optax
from optax import losses
from flax.training.train_state import TrainState

from gpt_lw.data import get_dataset, sample_batch
from gpt_lw.model_nanodo import DoConfig, TransformerDo

def train(
        dataset_path: str,
        n_steps: int,
        batch_size: int,
        seq_len: int,
        lr: float = 1e-3,
        seed: int = 42,
    ):

    # gen random keys
    key = jax.random.PRNGKey(seed)
    data_key, model_key = jax.random.split(key, 2)

    # device TODO check if this is correct
    device = jax.devices()[0]

    # load dataset
    ds, tokenizer = get_dataset(dataset_path)

    # init model
    config = DoConfig(
        D=16,
        H=4,
        L=64,
        N=4,
        V=tokenizer.vocab_size,
        F=64 * 2,
    )
    model = TransformerDo(config)

    inputs = jnp.ones((batch_size, seq_len), jnp.int32)
    params = model.init(model_key, inputs)

    optimizer = optax.adamw(lr)
    state = TrainState.create(apply_fn=model.apply, params=params['params'], tx=optimizer)

    def loss_fn(params, x, y):
        logits = state.apply_fn({'params': params}, x)
        loss = losses.softmax_cross_entropy_with_integer_labels(logits, y)
        return loss.mean()

    for step in range(n_steps):
        data_key, batch_key = jax.random.split(data_key)
        batch = sample_batch(ds, batch_size, seq_len, batch_key)
        x, y = batch[:, :-1], batch[:, 1:]

        loss = loss_fn(state.params, x, y)
        print("loss: ", loss)

        grads = jax.grad(loss_fn)(state.params, x, y)
        state = state.apply_gradients(grads=grads)

        print("batch: ", batch.shape)



if __name__ == "__main__":
    train(
        dataset_path="cfg_dataset/simple4_100000.txt",
        n_steps=100,
        batch_size=8,
        seq_len=5,
    )