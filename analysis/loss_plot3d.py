import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gpt_lw.model_utils import load_variables


if __name__ == "__main__":
    misc_metrics = load_variables("runs/tm_wiki_mini/checkpoints/last_misc_metrics.pkl")[0]
    Xs = jnp.concatenate([m[0] for m in misc_metrics])
    Ls = jnp.concatenate([m[2] for m in misc_metrics])

    print(Xs.shape, Ls.shape)  # both (32000, 512)
    # 100k steps / 2k steps per eval = 50 evals
    # reshape into per_step
    Xs = Xs.reshape(50, -1, Xs.shape[-1])
    Ls = Ls.reshape(50, -1, Ls.shape[-1])
    print(Xs.shape, Ls.shape)  # both (50, 640, 512)


    # first absolute position
    Ls = Ls.mean(axis=1)

    if False:
        for i in range(50):
            plt.plot(Ls[i])

            # plot i - 1 with a dimmer shade of the same color
            if i > 0:
                plt.plot(Ls[i - 1], alpha=0.5)
            if i > 1:
                plt.plot(Ls[i - 2], alpha=0.3)

            plt.title(f"step {i}")
            plt.show()

    if True: # 3d plot
        # Create a figure and a 3D axis
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for the x and y coordinates
        x = np.arange(Ls.shape[0])
        y = np.arange(Ls.shape[1])
        X, Y = np.meshgrid(x, y)

        # Plot the surface
        surf = ax.plot_surface(X.T, Y.T, Ls, cmap='viridis')

        # Set labels and title
        ax.set_xlabel('train steps')
        ax.set_ylabel('seq pos')
        ax.set_zlabel('loss')

        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Show the plot
        plt.show()



