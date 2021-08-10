import os
import contextlib
from functions import plot
import numpy as np
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def iteration(β, a, b, k, init_cond, alpha_gamma_mat, i, j):
    α, γ = alpha_gamma_mat[i, j]
    plot(α, β, γ, a, b, k, init_cond, i, j, base_path='FigImages')


def main():
    # lyapunov_exp_mat = np.load('Data/lyapunov_exp.npy')
    alpha_gamma_mat = np.load('Data/alpha_gamma.npy')

    # α y γ variable
    β = 1000.0
    a = -8 / 7
    b = -5 / 7
    k = 1
    init_cond = [1.1, 0.12, 0.01]

    if not os.path.isdir('FigImages'):
        os.makedirs('FigImages')

    # row_start = 600
    # row_end = lyapunov_exp_mat.shape[1]
    row_start = 0
    row_end = 1

    # col_end = lyapunov_exp_mat.shape[1]
    col_end = 10
    for i in range(row_start, row_end):
        with tqdm_joblib(tqdm(desc=f"Row: {i}", total=col_end)) as _:
            Parallel(n_jobs=12)(
                delayed(iteration)(β, a, b, k, init_cond, alpha_gamma_mat, i, j)
                for j in range(col_end))


if __name__ == '__main__':
    main()
