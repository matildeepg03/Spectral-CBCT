import astra
import numpy as np
from matplotlib import pyplot as plt

def astra_sino(N, ny, nz, angles, scale, source_origin, origin_detector, x, M, S):
    """
    Generate a sinogram using ASTRA toolbox for a phantom with multiple materials.

    Parameters:
    - N: int, size of the phantom (N x N x N)
    - ny: int, number of detector pixels in y-direction
    - nz: int, number of detector pixels in z-direction
    - n_proj: int, number of projections
    - angles: ndarray, array of projection angles
    - pitch: float, pixel spacing in mm
    - scale: float, scaling factor
    - source_origin: float, distance from source to origin in mm
    - origin_detector: float, distance from origin to detector in mm
    - depth: int, depth of the phantom
    - M: ndarray, attenuation matrix
    - S: ndarray, energy spectrum matrix

    Returns:
    - photon_counts: ndarray, generated photon counts
    or
    - noisy_counts: ndarray, generated photon counts with Poisson noise
    """

    #print("x shape:", x.shape)

    bone = x[:, 0].reshape((N, N, N))
    water = x[:, 1].reshape((N, N, N))

    # Bone
    # Create volume geometry
    lim = (N * scale) / 2
    vol_geom = astra.create_vol_geom(N, N, N, -lim, lim, -lim, lim, -lim, lim)
        
    # Create volume data in ASTRA
    volume_id_bone = astra.data3d.create('-vol', vol_geom, bone)
    volume_id_water = astra.data3d.create('-vol', vol_geom, water)

    # Create projection geometry
    proj_geom = astra.create_proj_geom('cone', scale, scale, nz, ny, angles, source_origin/10, origin_detector/10)
    
    # Generate sinogram using ASTRA's GPU implementation
    _, sinogram_bone = astra.create_sino3d_gpu(volume_id_bone, proj_geom, vol_geom)
    sinogram_bone = sinogram_bone.reshape((nz * len(angles) * ny))
    _, sinogram_water = astra.create_sino3d_gpu(volume_id_water, proj_geom, vol_geom)
    sinogram_water = sinogram_water.reshape((nz * len(angles) * ny))

    sinogram = np.column_stack([sinogram_bone, sinogram_water])
    np.save("sinogram_astra.npy", sinogram)
    attenuation_factor = np.exp(-M @ sinogram.T)

    photon_counts = S @ attenuation_factor
    np.save("photon_counts.npy", photon_counts)

    #noisy_counts = np.random.poisson(photon_counts) # COMMENT THIS LINE if you dont want noisy counts with the mask
    #mask = create_phantom_mask((nz, len(angles), ny))
    #mask = mask.ravel()
    #noisy_counts = photon_counts.copy()

    #noisy_counts[0, mask] = np.random.poisson(photon_counts[0, mask])
    #noisy_counts[1, mask] = np.random.poisson(photon_counts[1, mask])
    #np.save("noisy_counts_mask.npy", noisy_counts)
    return photon_counts
