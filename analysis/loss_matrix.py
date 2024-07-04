import os
import numpy as np
from tqdm import tqdm

from gpt_lw.data import TextTokenizer


def compute_mean_loss_rp_matrix(Xs, Ls, vocab_size):
    # Initialize the mean loss matrix
    mean_loss_rp = np.zeros((vocab_size, Xs.shape[1]))
    counter_rp = np.zeros((vocab_size, Xs.shape[1]))

    # Compute the mean loss matrix
    for k in tqdm(range(Xs.shape[0])):
        for i in range(Xs.shape[1]):
            for j in range(i, Xs.shape[1]):
                rp = j - i
                mean_loss_rp[Xs[k, i], rp] += Ls[k, j]
                counter_rp[Xs[k, i], rp] += 1

    mean_loss_rp = mean_loss_rp / counter_rp 
    mean_loss_rp[np.isnan(mean_loss_rp)] = 0  # replace NaNs with 0

    return mean_loss_rp


if __name__ == "__main__":
    # Load the data from the .npz file
    analysis_dir = "runs/tm_wiki_mini/analysis"
    data = np.load(os.path.join(analysis_dir, "history.npz"))
    tok = TextTokenizer()

    # Extract the arrays from the loaded data
    Xs = data['xt']
    Ls = data['loss']

    Xs = Xs[:20000]
    Ls = Ls[:20000]

    mean_loss_rp = compute_mean_loss_rp_matrix(Xs, Ls, tok.vocab_size)

    # Save the mean loss matrix
    np.save(os.path.join(analysis_dir, "mean_loss_rp.npy"), mean_loss_rp)


