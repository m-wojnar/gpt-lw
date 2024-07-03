import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gpt_lw.data import TextTokenizer

def sort_tokens_by_mean_loss(mean_loss_rp, range_start, range_end):
    # Calculate the average loss over the specified range
    average_loss_range = mean_loss_rp[:, range_start:range_end].mean(axis=1)
    # Sort indices based on the average loss in descending order
    sorted_indices = np.argsort(-average_loss_range)
    return sorted_indices

if __name__ == "__main__":
    # Load the mean loss matrix
    analysis_dir = "runs/tm_wiki_mini/analysis"
    mean_loss_rp = np.load(os.path.join(analysis_dir, "mean_loss_rp.npy"))
    tok = TextTokenizer()

    # Define the range for averaging the loss
    range_start = 0
    range_end = 5

    # Order the rows based on the average loss in the specified range
    sorted_indices = sort_tokens_by_mean_loss(mean_loss_rp, range_start, range_end)

    # Number of tokens to plot
    top_n = 10

    # Create a figure with subplots
    fig, axes = plt.subplots(top_n, 1, figsize=(10, 2 * top_n), sharex=True)

    for i in range(top_n):
        token_index = sorted_indices[i]
        token_str = tok.decode(jnp.array([token_index]))
        average_loss = mean_loss_rp[token_index]

        axes[i].plot(average_loss)
        axes[i].set_title(f"{token_str} (Index: {token_index})")
        axes[i].set_ylabel("Mean Loss")
    
    axes[-1].set_xlabel("Relative Position")
    
    plt.tight_layout()
    
    # Save the plot to a PNG file
    output_file = os.path.join(analysis_dir, "top_tokens_mean_loss.png")
    plt.savefig(output_file)
    
    # Show the plot
    plt.show()
