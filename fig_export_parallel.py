import os
from functions import plot
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def iteration(β, a, b, k, init_cond, alpha_gamma_mat, i, j):
    α, γ = alpha_gamma_mat[i, j]
    plot(α, β, γ, a, b, k, init_cond, i, j, base_path='FigImages')


def main():
    lyapunov_exp_mat = np.load('Data/lyapunov_exp.npy')
    alpha_gamma_mat = np.load('Data/alpha_gamma.npy')

    # α y γ variable
    β = 1000.0
    a = -8 / 7
    b = -5 / 7
    k = 1
    init_cond = [1.1, 0.12, 0.01]

    if not os.path.isdir('FigImages'):
        os.makedirs('FigImages')

    row_start = 600
    row_end = lyapunov_exp_mat.shape[0]

    for i in range(row_start, row_end):
        Parallel(n_jobs=8)(
            delayed(iteration)(β, a, b, k, init_cond, alpha_gamma_mat, i, j)
            for j in tqdm(range(lyapunov_exp_mat.shape[1])))


if __name__ == '__main__':
    main()
