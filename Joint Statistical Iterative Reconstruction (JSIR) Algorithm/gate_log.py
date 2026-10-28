import uproot
import numpy as np

nz, ny = 44, 64

z_min, z_max = -2.2, 2.2
y_min, y_max = -3.2, 3.2

E_thresh = 50.0

epsilon = 1e-3

min_counts = 5 

window_size = 10

n_proj = 360


open_low = np.load("low0_proj.npy")
open_high = np.load("high0_proj.npy")

phantom_low = np.load("low_proj2.npy")
phantom_high = np.load("high_proj2.npy")

def create_phantom_mask(sinogram_shape):
    """
    Create a binary mask for the phantom region.
    
    Parameters:
    - sinogram_shape: shape of the sinogram (e.g., from GATE: nz, nProj, ny)
    - phantom_region_coords: dict with bounds like {'z_min': 0, 'z_max': 10, 'y_min': 0, 'y_max': 20}
    
    Returns:
    - mask: binary array same shape as sinogram (1 for phantom, 0 for others)
    """

    
    z_min, z_max = 18, 26
    y_min, y_max = 17, 47
    

    nz, nProj, ny = sinogram_shape
    mask = np.zeros((nz, nProj, ny), dtype=bool)
    mask[z_min:z_max, :, y_min:y_max] = True

    
    return mask

def gate_air(n_proj):
    """
    Load and process photon counts from GATE simulation files (without phantom - I0 GATE).

    Returns:
    - gate_air: numpy array of shape (2, nPixels) containing the mean air sinograms for both energy bins
    """
    open_low = np.load("low0_proj.npy")
    open_high = np.load("high0_proj.npy")

    mean_air_low = np.mean(open_low, axis=1)    # (nz, ny)
    mean_air_high = np.mean(open_high, axis=1)  # (nz, ny)

    # Use the same air estimate for every phantom projection
    mean_air_low = np.repeat(mean_air_low[:, None, :], n_proj, axis=1)

    mean_air_high = np.repeat(mean_air_high[:, None, :], n_proj, axis=1)

    gate_air = np.stack([mean_air_low.reshape(-1), mean_air_high.reshape(-1)],axis=0)  # shape: (2, nPixels)

    gate_air_mean = np.array([np.mean(open_low), np.mean(open_high)])
    
    return gate_air, gate_air_mean  # shape: (2, nPixels)

mask_low_back = create_phantom_mask(open_low.shape)
mask_high_back = create_phantom_mask(open_high.shape)

_ , gate_air_mean = gate_air(n_proj)

open_low[~mask_low_back] = gate_air_mean[0]
open_high[~mask_high_back] = gate_air_mean[1]
phantom_low[~mask_low_back] = gate_air_mean[0]
phantom_high[~mask_high_back] = gate_air_mean[1]
print("gate_air shape:", gate_air_mean.shape)
print("open_low shape:", open_low.shape)
print("open_high shape:", open_high.shape)
print("phantom_low shape:", phantom_low.shape)
print("phantom_high shape:", phantom_high.shape)

atten_low_all = []
atten_high_all = []

for i in range(n_proj):
    print("test")
    indices = [(i + j) % n_proj for j in range(window_size)]

    matrix_I0_low = np.mean(open_low[:,indices,:], axis=1)
    matrix_I0_high = np.mean(open_high[:,indices,:], axis=1)

    matrix_low, matrix_high = phantom_low[:, i, :], phantom_high[:, i, :]

    atten_low = np.zeros((nz, ny), dtype=np.float32)
    atten_high = np.zeros((nz, ny), dtype=np.float32)

    mask_low = (matrix_I0_low > min_counts) & (matrix_low > 0)
    mask_high = (matrix_I0_high > min_counts) & (matrix_high > 0)

    atten_low[mask_low] = -np.log(matrix_low[mask_low] / matrix_I0_low[mask_low])

    atten_high[mask_high] = -np.log(matrix_high[mask_high] / matrix_I0_high[mask_high])

    atten_low = np.clip(atten_low, 0, 5)
    atten_high = np.clip(atten_high, 0, 5)

    atten_low_all.append(atten_low)
    atten_high_all.append(atten_high)

atten_low_stack = np.stack(atten_low_all, axis=1)

atten_high_stack = np.stack(atten_high_all, axis=1)


np.save("sino_low.npy", atten_low_stack)
np.save("sino_high.npy", atten_high_stack)