"""Utility functions for generating random elastic deformation maps."""

import numpy as np
from scipy.ndimage import (
    distance_transform_cdt,
    maximum_filter,
    map_coordinates,
)


def gaussian_kernel(size, sigma=1.):
    x = np.linspace(-1/sigma, 1/sigma, size)
    x *= x
    x *= -1/2
    np.exp(x, out=x)
    return x


def spherical_element(radius):
    s = slice(-radius, radius+1)
    grid = np.ogrid[s, s, s]
    for a in grid:
        a *= a
    return grid[0] + grid[1] + grid[2] <= radius * radius


def make_size_map(volume, max_perturbation: int = 15):
    """Create a map of the maximum perturbation for each region
    of the mask based on the size of mask elements.

    :param max_perturbation: The furthest distance to perturb in voxels.
    """

    #elem_size = 2*max_perturbation + 1
    #elem = np.empty((elem_size, elem_size, elem_size))
    
    size = distance_transform_cdt(volume)
    #maximum_filter(size, elem_size, output=size)
    maximum_filter(size, footprint=spherical_element(max_perturbation),
                   output=size)
    np.minimum(max_perturbation, size, out=size)
    return size


def perturb_image(volume, size_map, size_ratio: float = 1.2, nq: float = 0.03):
    """Randomly perturb a volume mask, where purturbation distance is
    indicated by `size_map`.

    :param size_ratio: Maximum fraction of object size to displace by.
    :param nq: Noise spatial frequency cutoff; higher varies more.
    """

    noise_vecs = np.empty((3,) + volume.shape)

    k0 = gaussian_kernel(volume.shape[0], nq)
    k1 = gaussian_kernel(volume.shape[1], nq)
    k2 = gaussian_kernel(volume.shape[2], nq)

    for i in range(3):
        noise = np.random.normal(size=volume.shape, scale=0.34)
        noise *= k0[:, None, None]
        noise *= k1[None, :, None]
        noise *= k2[None, None, :]
        noise = np.fft.fftshift(noise)
        #noise[0, 0, 0] = 0
        for j in range(3):
            noise = np.fft.ifft(noise, axis=j)
        noise = noise.real
        noise *= size_ratio * 0.34 / noise.std()
        noise *= size_map

        axis = [1, 1, 1]
        axis[i] = -1
        noise += np.arange(volume.shape[i]).reshape(axis)
        
        noise_vecs[i, :, :, :] = noise

    return map_coordinates(volume, noise_vecs, order=1, mode="nearest")


if __name__ == "__main__":
    import nrrd
    seg, _ = nrrd.read("annotations/annotation_082616_07_T40_iso_full.seg.nrrd",
        index_order="C")
    angio = 255 * seg[:, :, :, 1]
    size_map = make_size_map(angio)
    for i in range(10):
        out = perturb_image(angio, size_map)
        nrrd.write(f"perturbed{i}.nrrd", out, index_order="C")
