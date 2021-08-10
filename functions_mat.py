# NOT WORKING RIGHT NOW
import matplotlib.pyplot as plt
from typing import Tuple
from numba import njit
import numpy as np
from tqdm import tqdm
from contextlib import contextmanager
import joblib
from joblib import Parallel, delayed

from matplotlib import animation
from time import perf_counter


@njit
def transient(proximity: int, additive=300_000, scale=100_000) -> Tuple[int, int]:
    """
    Calculates the number of integration steps given the proximity to
    a bifurcation
    :param proximity: proximity to a bifurcation point between 1 and 10
    :param additive: additive term for the calculation
    :param scale: scale term for the calculation
    :returns n: number of integration steps
             t: number of transient steps to drop before considering a valid path
    :raise ValueError if proximity is not in range
    """
    if not (1 <= proximity <= 10):
        raise ValueError('Proximity out of range')

    n = additive + proximity * scale
    t = round(n*0.4)

    return n, t


@njit
def f(a: float, b: float, x: np.ndarray) -> float:
    """
    Piecewise linear function
    :param a: function variable
    :param b: function variable
    :param x: function variable
    :return: evaluated function
    """
    # return b*x + 0.5*(a-b)*(abs(x+1) - abs(x-1))
    return b*x + 0.5*(a-b)*(np.abs(x+1) - np.abs(x-1))


@njit
def dx_dt(a: float, b: float, k: float, α: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Derivative of the x coordinate wrt time
    :param a: function variable
    :param b: function variable
    :param k: function variable
    :param α: function variable
    :param x: function variable
    :param y: function variable
    :return: derivative evaluation
    """
    return k*α*(y - x - f(a, b, x))


@njit
def dy_dt(k: float, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Derivative of the y coordinate wrt time
    :param k: function variable
    :param x: function variable
    :param y: function variable
    :param z: function variable
    :return: derivative evaluation
    """
    return k*(x - y + z)


@njit
def dz_dt(k: float, β: float, γ: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Derivative of the z coordinate wrt time
    :param k: function variable
    :param β: function variable
    :param γ: function variable
    :param y: function variable
    :param z: function variable
    :return: derivative evaluation
    """
    return -k*(β*y + γ*z)


FRACTION = 1/6


@njit
def rk_solver(a: float, b: float, k: float,
              α: np.ndarray, β: float, γ: np.ndarray,
              x: np.ndarray, y: np.ndarray, z: np.ndarray,
              n_steps: int, transient_drop: int, trajectories: np.ndarray, h=0.001):
    """
    :param a: function variable
    :param b: function variable
    :param k: function variable
    :param α: function variable
    :param β: function variable
    :param γ: function variable
    :param x: initial condition
    :param y: initial condition
    :param z: initial condition
    :param n_steps: number of integration steps
    :param transient_drop: point of transient drop
    :param trajectories: matrix to store results
    :param h: integration step size
    """
    idx = 0
    for t in range(0, n_steps):
        k1 = dx_dt(a, b, k, α, x, y)
        l1 = dy_dt(k, x, y, z)
        m1 = dz_dt(k, β, γ, y, z)

        k2 = dx_dt(a, b, k, α, x + 0.5 * h * k1, y + 0.5 * h * l1)
        l2 = dy_dt(k, x + 0.5 * h * k1, y + 0.5 * h * l1, z + 0.5 * h * m1)
        m2 = dz_dt(k, β, γ, y + 0.5 * h * l1, z + 0.5 * h * m1)

        k3 = dx_dt(a, b, k, α, x + 0.5 * h * k2, y + 0.5 * h * l2)
        l3 = dy_dt(k, x + 0.5 * h * k2, y + 0.5 * h * l2, z + 0.5 * h * m2)
        m3 = dz_dt(k, β, γ, y + 0.5 * h * l2, z + 0.5 * h * m2)

        k4 = dx_dt(a, b, k, α, x + h * k3, y + h * l3)
        l4 = dy_dt(k, x + h * k3, y + h * l3, z + h * m3)
        m4 = dz_dt(k, β, γ, y + h * l3, z + h * m3)

        x = x + FRACTION * (k1 + 2 * k2 + 2 * k3 + k4) * h
        y = y + FRACTION * (l1 + 2 * l2 + 2 * l3 + l4) * h
        z = z + FRACTION * (m1 + 2 * m2 + 2 * m3 + m4) * h

        if t >= transient_drop:
            trajectories[:, idx, 0] = x
            trajectories[:, idx, 1] = y
            trajectories[:, idx, 2] = z
            idx += 1


@contextmanager
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


def plot_iteration(trajectories: np.ndarray, row: int, col: int, base_path: str, elevation: int, azimuth: int):
    # ax = plt.axes(projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.view_init(elev=elevation, azim=azimuth)
    ax.plot3D(trajectories[col, :, 0], trajectories[col, :, 1], trajectories[col, :, 2], 'black', linewidth=0.5)

    if base_path:
        fig.savefig(f'{base_path}/{row}_{col}.png', bbox_inches='tight')
        fig.clf()
        # plt.close()
    else:
        fig.show()


def plot(α: np.ndarray, β: float, γ: np.ndarray,
         a: float, b: float, k: float,
         x_init: np.ndarray, y_init: np.ndarray, z_init: np.ndarray, trajectories: np.ndarray,
         n_steps: int, transient_drop: int, row: int, base_path=None, n_jobs=1):
    print(f'Solving batch of {trajectories.shape[0]} ODEs for row: {row}')
    rk_solver(a, b, k, α, β, γ, x_init, y_init, z_init, n_steps, transient_drop, trajectories)

    # elevation = np.random.randint(0, 180)
    # azimuth = np.random.randint(0, 360)
    elevation, azimuth = 45, 45

    with tqdm_joblib(tqdm(desc=f"Row: {row}", total=trajectories.shape[0])) as _:
        Parallel(n_jobs=n_jobs)(
            delayed(plot_iteration)(trajectories, row, col, base_path, elevation, azimuth)
            for col in range(trajectories.shape[0]))


# NOT WORKING
# def update(num, data, line):
#     line.set_data(data[num, :, :2].T)
#     line.set_3d_properties(data[num, :, 2])
#     return line,
#
#
# def plot_animation(α: np.ndarray, β: float, γ: np.ndarray,
#          a: float, b: float, k: float,
#          x_init: np.ndarray, y_init: np.ndarray, z_init: np.ndarray, trajectories: np.ndarray,
#          n_steps: int, transient_drop: int, row: int, base_path=None, n_jobs=1):
#     print(f'Solving batch of {trajectories.shape[0]} ODEs for row: {row}')
#     start = perf_counter()
#     rk_solver(a, b, k, α, β, γ, x_init, y_init, z_init, n_steps, transient_drop, trajectories)
#     end = perf_counter()
#     print(f'Took: {end-start} seconds')
#     elevation, azimuth = 45, 45
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_axis_off()
#     ax.view_init(elev=elevation, azim=azimuth)
#     print('Generating animation')
#     start = perf_counter()
#     line, = ax.plot3D(trajectories[0, :, 0], trajectories[0, :, 1], trajectories[0, :, 2], 'black', linewidth=0.5)
#     ani = animation.FuncAnimation(fig, update, trajectories.shape[0], fargs=(trajectories, line),
#                                   interval=1/trajectories.shape[1], blit=True)
#                                   # repeat = False, interval = 1 / trajectories.shape[0], blit = True)
#     end = perf_counter()
#     print(f'Took: {end - start} seconds')
#
#     print('Saving animation')
#     start = perf_counter()
#     ani.save(f'{base_path}/{row}.png', writer='imagemagick')
#     end = perf_counter()
#     print(f'Took: {end - start} seconds')
#
#
# def plot_blit(α: np.ndarray, β: float, γ: np.ndarray,
#          a: float, b: float, k: float,
#          x_init: np.ndarray, y_init: np.ndarray, z_init: np.ndarray, trajectories: np.ndarray,
#          n_steps: int, transient_drop: int, row: int, base_path=None, n_jobs=1):
#     print(f'Solving batch of {trajectories.shape[0]} ODEs for row: {row}')
#     start = perf_counter()
#     rk_solver(a, b, k, α, β, γ, x_init, y_init, z_init, n_steps, transient_drop, trajectories)
#     end = perf_counter()
#     print(f'Took: {end-start} seconds')
#     elevation, azimuth = 45, 45
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_axis_off()
#     ax.view_init(elev=elevation, azim=azimuth)
#     line, = ax.plot3D(trajectories[0, :, 0], trajectories[0, :, 1], trajectories[0, :, 2], 'black', linewidth=0.5,
#                       animated=True)
#
#     plt.show(block=False)
#     plt.pause(0.1)
#     plt.close()
#
#     bg = fig.canvas.copy_from_bbox(fig.bbox)
#     ax.draw_artist(line)
#     fig.canvas.blit(fig.bbox)
#
#     for col in range(trajectories.shape[0]):
#         fig.canvas.restore_region(bg)
#         line.set_data(trajectories[col, :, :2].T)
#         line.set_3d_properties(trajectories[col, :, 2])
#         ax.draw_artist(line)
#         fig.canvas.blit(fig.bbox)
#         fig.canvas.flush_events()
#
#         fig.savefig(f'{base_path}/{row}_{col}.png', bbox_inches='tight')
