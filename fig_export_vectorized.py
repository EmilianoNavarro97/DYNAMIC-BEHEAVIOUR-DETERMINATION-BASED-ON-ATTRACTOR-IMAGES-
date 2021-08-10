# NOT WORKING RIGHT NOW
import os

import matplotlib.pyplot as plt

from functions_mat import plot, transient
import numpy as np
# from tqdm import tqdm


def main():
    print(plt.get_backend())
    alpha_gamma_mat = np.load('Data/alpha_gamma.npy')

    # User params
    n_batch = 12
    # n_batch = 10
    # n_rows = alpha_gamma_mat.shape[0]
    n_rows = 1

    # α y γ variable
    β = 1000.0
    a = -8 / 7
    b = -5 / 7
    k = 1
    init_cond = [1.1, 0.12, 0.01]

    if not os.path.isdir('FigImages'):
        os.makedirs('FigImages')

    n_steps, transient_drop = transient(1)
    x_init = np.array([init_cond[0]] * n_batch)
    y_init = np.array([init_cond[1]] * n_batch)
    z_init = np.array([init_cond[2]] * n_batch)
    trajectories = np.empty((n_batch, n_steps - transient_drop, 3), dtype=np.float32)
    for i in range(n_rows):
        crop = alpha_gamma_mat[0, :n_batch]
        α = crop[:, 0]
        γ = crop[:, 1]
        plot(α, β, γ, a, b, k, x_init, y_init, z_init, trajectories, n_steps, transient_drop, i, base_path='FigImages',
             n_jobs=12)


if __name__ == '__main__':
    main()
