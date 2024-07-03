import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from gpt_lw.loss import compute_relative_positions
from gpt_lw.data import TextTokenizer
from gpt_lw.model_utils import load_variables

from transformers import AutoTokenizer

enc = TextTokenizer()
# enc = AutoTokenizer.from_pretrained("gpt2")


# DELIM_TOKEN = enc.encode("\n")
# DELIM_TOKEN = enc.encode(".\n")
# DELIM_TOKEN = enc.encode("\n ")
# DELIM_TOKEN = enc.encode(".")
# DELIM_TOKEN = enc.encode(" that")
# DELIM_TOKEN = enc.encode(" someone")
# DELIM_TOKEN = enc.encode(" the")

# DELIM_TOKEN = enc.encode("\n\n")
DELIM_TOKEN = enc.encode("<|endoftext|>")

assert len(DELIM_TOKEN) == 1, DELIM_TOKEN
DELIM_TOKEN_STR = enc.decode(DELIM_TOKEN)
DELIM_TOKEN_STR = DELIM_TOKEN_STR.replace("\n", "\\n")
DELIM_TOKEN = DELIM_TOKEN[0]
print(f"delim token ({DELIM_TOKEN_STR}): ", DELIM_TOKEN)

# DELIM_TOKEN = -1
# DELIM_TOKEN_STR = "testing sth"

N_RP = 512
SEQ_LEN = 512

def compute_avg_loss_per_token():
    # _, _, misc_metrics, _ = load_variables("../runs/debug/checkpoints/last.pkl")
    _, _, misc_metrics, _ = load_variables("runs/tm_wiki_mini_old/checkpoints/last.pkl")
    # _, _, misc_metrics, _ = load_variables("runs/tt_wiki_mini/checkpoints/last.pkl")
    Xs = jnp.concatenate([m[0] for m in misc_metrics])
    Ls = jnp.concatenate([m[2] for m in misc_metrics])
    # Gs = jnp.concatenate([m[3] for m in misc_metrics])

    # Load the data from the .npz file
    # data = np.load("analysis/history.npz")
    # # Extract the arrays from the loaded data
    # Xs = jnp.array(data['xt'])
    # Ls = jnp.array(data['loss'])

    print(f"Xs: {Xs.shape}, Ls: {Ls.shape}")

    # last N steps
    Xs = Xs[-10000:]
    Ls = Ls[-10000:]

    # cut off first half of sequence (axis=1)
    Xs = Xs[:, 128:-128]
    Ls = Ls[:, 128:-128]

    print(f"Xs: {Xs.shape}, Ls: {Ls.shape}")

    # check if DELIM_TOKEN in Xs
    dt_mask = Xs == DELIM_TOKEN
    dt_mask = jnp.any(dt_mask, axis=1)
    # # remove rows which don't have dt_mask == 1
    Xs = Xs[dt_mask]
    Ls = Ls[dt_mask]

    if Xs.shape[0] == 0:
        print("DELIM_TOKEN not found in Xs")
        return

    n = Xs.shape[0]

    Xs = Xs[:n]

    RPs = compute_relative_positions(Xs, delim_token=DELIM_TOKEN)

    # # array for each relative position value (shape (N_RP,))
    # count = [0] * N_RP
    # loss_sum = [0] * N_RP
    # # compute how many tokens have each relative position and sum the loss for each position
    # for i in tqdm(range(Xs.shape[0])):
    #     for j in range(Xs.shape[1]):
    #         rp = RPs[i, j]
    #         l = Ls[i, j]
    #         count[rp] += 1
    #         loss_sum[rp] += l.item()
    # compute average
    # avg_loss = [loss_sum[i] / count[i] if count[i] != 0 else 0 for i in range(N_RP)]

    # instead of summing and keeping count, just append
    losses = [[] for _ in range(N_RP)]
    for i in tqdm(range(Xs.shape[0])):
        for j in range(Xs.shape[1]):
            rp = RPs[i, j]
            l = Ls[i, j]
            losses[rp].append(l.item())

    # compute average and variance
    avg_loss = [np.mean(losses[i]) if len(losses[i]) != 0 else 0 for i in range(N_RP)]
    var_loss = [np.var(losses[i]) if len(losses[i]) != 0 else 0 for i in range(N_RP)]

    # save as numpy array
    # np.save("../runs/wiki_mini/analysis/avg_loss.npy", np.array(avg_loss))

    # matplotlib code to plot the average loss and variance loss in two panels
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(avg_loss)
    axs[0].set_title('mean loss')

    axs[1].plot(var_loss)
    axs[1].set_title('var loss')

    fig.suptitle(DELIM_TOKEN_STR)
    plt.show()

    # plt.plot(avg_loss)
    # plt.title(DELIM_TOKEN_STR)
    # plt.show()

def convert_to_pdist(avg_loss_per_token):
    weights = avg_loss_per_token
    weights = np.array(weights)

    # normalize weights so its prob dist
    weights = weights / np.sum(weights)
    # invert so higher loss -> lower weight
    weights = 1.0 - weights
    weights = weights / np.sum(weights)

    # integral = seq_len
    weights = weights * SEQ_LEN
    return weights

def convert_to_max_minus_loss(avg_loss_per_token):
    weights = 11.413 - avg_loss_per_token
    weights = weights / np.sum(weights)
    weights = weights * SEQ_LEN
    return weights

def adaptive_moving_average(data, max_window_size):
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        window_size = min(i + 1, max_window_size)
        smoothed_data[i] = np.mean(data[max(0, i - window_size + 1):i + 1])
    return smoothed_data

def convert_to_max_minus_loss_pow_smoothed(avg_loss_per_token, max_window_size=15):
    # Apply adaptive moving average to smooth the data
    avg_loss_per_token = adaptive_moving_average(avg_loss_per_token, max_window_size)

    # The rest of the original function
    weights = 11.413 - avg_loss_per_token
    weights = weights / np.sum(weights)
    weights = weights ** 4
    weights = weights / np.sum(weights)
    weights = weights * SEQ_LEN

    print(np.sum(weights))
    return weights


if __name__ == "__main__":

    compute_avg_loss_per_token()


    '''
    avg_loss = np.load("../runs/wiki_mini/analysis/avg_loss.npy")
    avg_loss = avg_loss[:N_RP]

    # weights_base = convert_to_pdist(avg_loss)
    weights = convert_to_max_minus_loss(avg_loss)
    # weights = convert_to_max_minus_loss_pow_smoothed(avg_loss, max_window_size=30)
    # plt.plot(weights_base)
    plt.plot(weights)
    plt.show()

    print(weights.shape)

    # np.save("../runs/wiki_mini/analysis/weights_pow_smoothed.npy", weights)
    '''









