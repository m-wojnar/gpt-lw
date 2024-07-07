import os
import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def convert_to_max_minus_loss(avg_loss_per_token, max_loss, seq_len):
    weights = max_loss - avg_loss_per_token
    weights = weights / np.sum(weights)
    weights = weights * seq_len
    return weights

def adaptive_moving_average(data, max_window_size):
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        window_size = min(i + 1, max_window_size)
        smoothed_data[i] = np.mean(data[max(0, i - window_size + 1):i + 1])
    return smoothed_data

def fit_func(y, extrapolate_seq_len, func="power_law"):
    x = np.arange(len(y))

    # Define the power law function
    def power_law(x, a, b, c):
        return a * np.power(x, b) + c
    def exponential(x, a, b, c):
        return a * np.exp(-(b * x)) + c
    
    func = power_law if func == "power_law" else exponential

    # Fit the power law to the data
    params, covariance = curve_fit(func, x, y)

    # Extract the fitted parameters
    a, b, c = params
    # Print the fitted parameters
    print(f'Fitted parameters: a = {a}, b = {b} c = {c}')

    # Generate fitted values
    if len(x) < extrapolate_seq_len:
        x = np.arange(extrapolate_seq_len)
    y_fit = func(x, a, b, c)

    return y_fit

if __name__ == "__main__":
    analysis_dir = "runs/llama_wiki_mini/analysis"
    data = np.load(os.path.join(analysis_dir, "loss_pos.npz"))

    init_loss = 10.8
    avg_loss = 4.213

    if True:  # abs_pos
        mean_Ls = data["abs"]
        mean_Ls = adaptive_moving_average(mean_Ls, 10)
        mean_Ls = convert_to_max_minus_loss(mean_Ls, init_loss, seq_len=512)
        plt.plot(mean_Ls)
        plt.show()

        # save to analysis_dir
        np.save(os.path.join(analysis_dir, "abs_weights.npy"), mean_Ls)
    
    if True:  # rel_pos
        # mean_Ls = data["rel_16"]
        mean_Ls = data["rel_0"]

        mean_Ls = np.abs(mean_Ls - mean_Ls.mean())
        mean_Ls = adaptive_moving_average(mean_Ls, 15)

        mean_Ls = mean_Ls[:200]

        mean_Ls = fit_func(mean_Ls, extrapolate_seq_len=512, func="exponential")
        mean_Ls = convert_to_max_minus_loss(mean_Ls, init_loss, seq_len=512)

        plt.plot(mean_Ls)
        plt.show()

        # save to analysis_dir
        np.save(os.path.join(analysis_dir, "rel_weights.npy"), mean_Ls)

