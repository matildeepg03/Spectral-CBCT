import numpy as np
from matplotlib import pyplot as plt

def load_gatesino(gate1, gate2):
    """
    Load photon counts from GATE simulation files and return them as numpy arrays.

    Parameters:
    - gate1: str, path to the first GATE sinogram file (e.g., low energy)
    - gate2: str, path to the second GATE sinogram file (e.g., high energy)

    Returns:
    - tuple of two numpy arrays representing the sinograms
    """
    gate1 = np.asarray(gate1, dtype=np.float64)
    gate2 = np.asarray(gate2, dtype=np.float64)

    # GATE export shape: (nz, nProj, ny)
    nz, nProj, ny = np.shape(gate1)

    nPixels = nz * nProj * ny
    sinogram1 = gate1.reshape(nPixels)
    sinogram2 = gate2.reshape(nPixels)

    sinogram = np.column_stack([sinogram1, sinogram2])
    sinogram = sinogram.transpose(1, 0)

    return sinogram
