import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gpt_lw.data import TextTokenizer

def sort_tokens_by_mean_loss(mean_loss_rp, range_start, range_end, top_n):
    # Calculate the average loss over the specified range
    average_loss_range = mean_loss_rp[:, range_start:range_end].mean(axis=1)
    # Sort indices based on the average loss in descending order
    sorted_indices = np.argsort(-average_loss_range)
    return sorted_indices[:top_n]

def select_specific_tokens(tok, strings):
    enc_strs = []
    for string in strings:
        es = tok.encode(string)
        assert len(es) == 1
        enc_strs.append(es[0])
    return enc_strs


def plot_mean_loss(mean_loss_rp, indices, tok, output_file, separate_panels=True):
    # Number of tokens to plot
    num_tokens = len(indices)

    if separate_panels:
        # Create a figure with subplots
        fig, axes = plt.subplots(num_tokens, 1, figsize=(10, 2 * num_tokens), sharex=True)

        # If there's only one subplot, axes is not a list
        if num_tokens == 1:
            axes = [axes]

        for i, token_index in enumerate(indices):
            token_str = tok.decode(jnp.array([token_index]))
            token_str = token_str.replace("\n", "\\n")
            average_loss = mean_loss_rp[token_index]

            axes[i].plot(average_loss)
            axes[i].set_title(f"'{token_str}' (index: {token_index})")
            axes[i].set_ylabel("mean loss")
        
        axes[-1].set_xlabel("relative position")

    else:
        # Create a single plot
        plt.figure(figsize=(10, 6))
        for token_index in indices:
            token_str = tok.decode(jnp.array([token_index]))
            token_str = token_str.replace("\n", "\\n")
            average_loss = mean_loss_rp[token_index]

            plt.plot(average_loss, label=f"'{token_str}' (index: {token_index})")
        
        plt.xlabel("relative position")
        plt.ylabel("mean loss")
        plt.legend()
        plt.title("Mean Loss for Selected Tokens")

    plt.tight_layout()
    
    # Save the plot to a PNG file
    plt.savefig(output_file)
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Load the mean loss matrix
    analysis_dir = "runs/tm_wiki_mini/analysis"
    mean_loss_rp = np.load(os.path.join(analysis_dir, "mean_loss_rp.npy"))
    tok = TextTokenizer()

    # Selection function: Change this to use different selection criteria
    # selection_function = lambda: sort_tokens_by_mean_loss(mean_loss_rp, 1, 5, 5)
    selection_function = lambda: select_specific_tokens(tok, ["<|endoftext|>", "\n\n", "\n", ".\n", ".", ","])

    # Get the indices to plot
    selected_indices = selection_function()

    # Output file for the plot
    output_file = os.path.join(analysis_dir, "selected_tokens_mean_loss.png")

    # Plot and save the mean loss for the selected indices
    plot_mean_loss(mean_loss_rp, selected_indices, tok, output_file, separate_panels=True)
