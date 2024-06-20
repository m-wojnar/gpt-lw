import jax
from gpt_lw.data import get_dataset, sample_batch
from gpt_lw.gpt import GPT

def train(
        dataset_path: str,
        n_steps: int,
        batch_size: int,
        seq_len: int,
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
    # TODO

    for step in range(n_steps):
        data_key, batch_key = jax.random.split(data_key)
        batch = sample_batch(ds, batch_size, seq_len, batch_key)

        print(step, batch)



if __name__ == "__main__":
    # TODO: get config
    # TODO: pass into main
    train(
        dataset_path="cfg_dataset/simple4_100000.txt",
        n_steps=10,
        batch_size=2,
        seq_len=5,
    )