import jax
import jax.numpy as jnp
import optax
from optax import losses
from flax.training.train_state import TrainState

from gpt_lw.data import get_dataset, sample_batch
from gpt_lw.model_nanodo import DoConfig, TransformerDo
from gpt_lw.loss import get_weighted_loss

def train(
        dataset_path: str,
        n_steps: int,
        batch_size: int,
        seq_len: int,
        loss_weighting: str = "unweighted",
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

    loss_fn = get_weighted_loss(state, loss_weighting)

    for step in range(n_steps):
        data_key, batch_key = jax.random.split(data_key)
        batch = sample_batch(ds, batch_size, seq_len, batch_key)
        xt, xtp1 = batch[:, :-1], batch[:, 1:]

        loss = loss_fn(state.params, xt, xtp1)
        print("loss: ", loss)

        grads = jax.grad(loss_fn)(state.params, xt, xtp1)
        state = state.apply_gradients(grads=grads)


if __name__ == "__main__":
    train(
        dataset_path="cfg_dataset/simple4_100000.txt",
        n_steps=10,
        batch_size=8,
        seq_len=5,
        loss_weighting="negexp_relpos",
    )