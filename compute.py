import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from gpt_lw.loss import compute_relative_positions

DELIM_TOKEN = 0
N_RP = 27


if __name__ == "__main__":
    # load Xs, Ys, and Ls
    Xs = jnp.load("runs/base_unweighted/analysis/Xs.npy")
    Ys = jnp.load("runs/base_unweighted/analysis/Ys.npy")
    Ls = jnp.load("runs/base_unweighted/analysis/Ls.npy")
    print(f"Xs: {Xs.shape}, Ys: {Ys.shape}, Ls: {Ls.shape}")

    # n = Xs.shape[0]
    n = 10000

    RPs = compute_relative_positions(Xs, delim_token=DELIM_TOKEN)

    # array for each relative position value (shape (N_RP,))
    count = [0] * N_RP
    loss_sum = [0] * N_RP

    # compute how many tokens have each relative position and sum the loss for each position
    for i in tqdm(range(n)):
        for j in range(Xs.shape[1]):
            rp = RPs[i, j]
            l = Ls[i, j]
            count[rp] += 1
            loss_sum[rp] += l.item()
        
    print(count)
    print(loss_sum)
    # compute average
    avg_loss = [loss_sum[i] / count[i] if count[i] != 0 else 0 for i in range(N_RP)]
    # matplotlib code to plot the loss for each relative position
    plt.plot(avg_loss)
    plt.show()









