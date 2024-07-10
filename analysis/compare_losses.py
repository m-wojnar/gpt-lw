import jax.numpy as jnp
import matplotlib.pyplot as plt

from gpt_lw.loss import compute_relative_positions


if __name__ == '__main__':
    unweighted_model = 'llama_wiki_mini_short'

    for weighted_model in [
        'llama_wiki_mini_short_abs', 'llama_wiki_mini_short_abs_4', 'llama_wiki_mini_short_abs_8',
        'llama_wiki_mini_short_rel', 'llama_wiki_mini_short_rel_4', 'llama_wiki_mini_short_rel_8',
        'llama_wiki_mini_short_rel_nn', 'llama_wiki_mini_short_abs_random'
    ]:
        w_name = weighted_model.split('_')[4:]

        unw_history = jnp.load(f'{unweighted_model}_history.npz')
        unw_xtp1, unw_loss = jnp.asarray(unw_history['xtp1']), jnp.asarray(unw_history['loss'])

        w_history = jnp.load(f'{weighted_model}_history.npz')
        w_xtp1, w_loss = jnp.asarray(w_history['xtp1']), jnp.asarray(w_history['loss'])

        assert jnp.allclose(unw_xtp1, w_xtp1)
        delim_token = 0 if 'rel_nn' in weighted_model else 16

        if 'abs' in weighted_model:
            positions = jnp.arange(w_xtp1.shape[1])[None, :]
            positions = jnp.repeat(positions, w_xtp1.shape[0], axis=0)
        elif 'rel' in weighted_model:
            mask = (w_xtp1 == delim_token).any(axis=1)
            w_xtp1, w_loss = w_xtp1[mask], w_loss[mask]
            unw_xtp1, unw_loss = unw_xtp1[mask], unw_loss[mask]
            assert jnp.allclose(unw_xtp1, w_xtp1)
            positions = compute_relative_positions(w_xtp1, delim_token)
        else:
            raise ValueError('Not supported')

        first_idx = jnp.arange(positions.shape[0])[:, None]
        w_loss = w_loss[first_idx, positions].mean(axis=0)
        unw_loss = unw_loss[first_idx, positions].mean(axis=0)

        plt.plot(unw_loss - w_loss)
        plt.plot(jnp.zeros_like(unw_loss))

        if weighted_model == 'abs_random':
            random_weights = jnp.load('runs/llama_wiki_mini/analysis/abs_random_weights.npy')[:positions.shape[1]]
            mask = (random_weights < 1)
            xs = jnp.arange(mask.shape[0])[mask]
            ys = unw_loss[mask] - w_loss[mask]
            plt.scatter(xs, ys, c='red', marker='x', s=5)

            unmask = jnp.logical_not(mask)
            xs = jnp.arange(unmask.shape[0])[unmask]
            ys = unw_loss[unmask] - w_loss[unmask]
            plt.scatter(xs, ys, c='green', marker='x', s=5)

        plt.xlabel(('Absolute' if 'abs' in weighted_model else 'Relative') + ' position')
        plt.ylabel('Difference in loss')
        plt.ylim(-1, 1)
        plt.title(f'unw - {w_name}' + (f' [delim token = {delim_token}]' if 'rel' in weighted_model else ''))
        plt.tight_layout()
        plt.savefig(f'unw_{w_name}_loss_diff.png')
        plt.show()
