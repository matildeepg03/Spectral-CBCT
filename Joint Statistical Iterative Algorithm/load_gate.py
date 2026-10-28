import numpy as np
from matplotlib import pyplot as plt

def load_gate(gate1, gate2):
    """
    Load sinograms from GATE simulation files and return them as numpy arrays.

    Parameters:
    - gate1: str, path to the first GATE sinogram file
    - gate2: str, path to the second GATE sinogram file

    Returns:
    - tuple of two numpy arrays representing the sinograms
    """

    # Load the sinograms from the provided GATE files
    sinogram1 = np.load(gate1)
    sinogram2 = np.load(gate2)

    nProj, ny, nz = np.shape(sinogram1)
    nPixels = nz * nProj * ny

    sinogram = np.column_stack([sinogram1.reshape(nPixels), sinogram2.reshape(nPixels)])
    print("Shape of sinogram:", sinogram.shape)
    np.save("y_gate.npy", sinogram)
    return sinogram