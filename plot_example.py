import numpy as np

from scipy.stats import dirichlet
from barycentric import create_regular_polygon, barycentric, plot_density


def rot_mat(angle: float) -> np.ndarray:
    """Matrix for rotation through an `angle` radians.

    :param angle: rotation angle in radians
    :return: 2D rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


if __name__ == "__main__":

    m = 6
    alpha_vec = np.ones(m)
    alpha_vec[1] = 2
    alpha_vec[-1] = 2
    alpha_vec[3] = 2

    k = 10000
    x = np.linspace(-0.9, 0.9, int(np.sqrt(k)))
    y = np.linspace(-0.9, 0.9, int(np.sqrt(k)))

    mesh = np.asarray(np.meshgrid(x, y)).T.reshape(-1, 2)
    mesh = mesh[np.hypot(mesh[:, 0], mesh[:, 1]) < 2]

    polygon = create_regular_polygon(m)

    bary_mesh = np.asarray([barycentric(polygon, z) for z in mesh])
    msk = (bary_mesh >= 0).all(axis=1)
    bary_mesh = bary_mesh[msk]

    data = np.hstack(
        [bary_mesh, np.apply_along_axis(lambda x: dirichlet.pdf(x, alpha_vec), 1, bary_mesh).reshape(-1, 1)]
    )

    plot_density(data)
