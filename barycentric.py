import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import Axes
from typing import Optional, List


def create_regular_polygon(n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("n must be greater than 2")
    angles = 2 * np.pi * np.linspace(0, 1, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    return np.vstack([x, y]).T


def plot_polygon(poly: np.ndarray, ax: Optional[Axes] = None) -> None:
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(poly[range(-1, len(poly)), 0], poly[range(-1, len(poly)), 1])
    ax.set_aspect("equal")

    if ax is None:
        plt.show()
        plt.close()


def barycentric(poly: np.ndarray, x: List[float]) -> np.ndarray:
    tan_list = []
    norm_list = []
    for v, v_next in zip(poly[range(-1, len(poly))], poly):
        norm = (v - x) @ (v - x)
        norm_list.append(norm)

        e = (v - x) / norm
        e_next = (v_next - x) / ((v_next - x) @ (v_next - x))

        cs = e @ e_next
        sn = np.cross(e, e_next)
        tan = (1 - cs) / sn

        tan_list.append(tan)

    w = np.roll(np.fromiter(((tan_list[i - 1] + tan_list[i]) / norm_list[i] for i in range(len(poly))), dtype=float), 3)
    return w / w.sum()


def inverse_barycentric(poly: np.ndarray, phi: np.ndarray) -> np.ndarray:
    if phi.ndim != 2:
        raise ValueError("phi should be 2-dimensional")
    if phi.shape[1] != len(poly):
        raise ValueError("phi should have same second dimension as number of vertices in poly")
    return np.tensordot(phi, poly, axes=1)


def plot_density(arr: np.ndarray, levels: int = 10):
    n = arr.shape[1] - 1
    coord = arr[:, :-1]

    poly = create_regular_polygon(n)
    points = inverse_barycentric(poly, coord)
    x = points[:, 0].flatten()
    y = points[:, 1].flatten()
    z = arr[:, -1].flatten()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.tricontour(x, y, z, levels=levels)
    ax.tricontourf(x, y, z, levels=levels)

    plot_polygon(poly, ax)
