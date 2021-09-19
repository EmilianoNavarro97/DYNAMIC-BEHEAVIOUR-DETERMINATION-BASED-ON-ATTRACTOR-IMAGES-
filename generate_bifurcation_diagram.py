import os
from functions import save_x, tqdm_joblib
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def iteration(α_space, β, γ, a, b, k, init_cond, col):
    save_x(α_space[col], β, γ, a, b, k, init_cond, col)


def main():
    # α_space = np.linspace(6.2054, 8.2467, 1200)
    α_space = np.linspace(6.2054, 8.2467, 1200)
    β = 10.8976626192
    γ = -0.0447440294
    a = -1.1819730746
    b = -0.6523354182
    k = 1
    init_cond = [0.02, 0.01, 0.0]

    with tqdm_joblib(tqdm(desc=f"Generating X", total=len(α_space))) as _:
        Parallel(n_jobs=12)(
            delayed(iteration)(α_space, β, γ, a, b, k, init_cond, j)
            for j in range(len(α_space)))


if __name__ == '__main__':
    main()
