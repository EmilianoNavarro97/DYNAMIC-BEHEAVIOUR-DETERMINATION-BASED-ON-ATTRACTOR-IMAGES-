import os
from functions import plot
import numpy as np
from tqdm import tqdm


def main():
    alpha_gamma_mat = np.load('Data/alpha_gamma.npy')

    # α y γ variable
    β = 1000.0
    a = -8 / 7
    b = -5 / 7
    k = 1
    init_cond = [1.1, 0.12, 0.01]

    if not os.path.isdir('FigImages'):
        os.makedirs('FigImages')

    # for i in range(lyapunov_exp_mat.shape[0]):
    #     for j in tqdm(range(lyapunov_exp_mat.shape[1])):
    for i in range(1):
        for j in range(1):
            α, γ = alpha_gamma_mat[i, 99]
            plot(α, β, γ, a, b, k, init_cond, i, j)


if __name__ == '__main__':
    main()
