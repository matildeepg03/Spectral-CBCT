import matplotlib.pyplot as plt
import numpy as np
from simulatepreps_phantom import simulatepreps
from phantom import phantom_print
"""
The functions in this file are taken and slightly adjusted from the Jupyter notebook collectData.ipynb from BEP Luuk Froling.
"""
def getSinogram(y, astra, energy_bin_index=0, z_index=32, nProj=100, detector_shape=(64, 44)):
    nY, nZ = detector_shape
    y_bin = y[energy_bin_index]
    if astra == 0:
        y_proj_zy = y_bin.reshape((nProj, nZ, nY))  # shape: (100, 64, 44)
        sinogram = y_proj_zy[:, z_index, :]  # shape: (100, 64)
    else:
        y_proj_zy = y_bin.reshape((nZ, nProj, nY))
        sinogram = y_proj_zy[z_index, :, :]
        sinogram = sinogram[:,::-1]# Take slice and transpose to get (nProj, nY)
    return sinogram


def showSinogram(y, name, type =1, energy_bin_index=0, z_index=32, nProj=100, detector_shape=(64, 44)):
    # nX, nZ = detector_shape
    # y_bin = y[energy_bin_index]
    # y_proj_zx = y_bin.reshape((nProj, nZ, nX))  # shape: (100, 64, 44)
    sinogram = getSinogram(y, astra, energy_bin_index, z_index, nProj, detector_shape)  # shape: (100, 44)
    print(sinogram.shape)
    I0 = np.max(sinogram)
    attenuation = -np.log((sinogram + 1) / (I0 + 1))
    if type ==0:
        plt.figure(figsize=(6, 4))
        plt.imshow(sinogram, aspect='auto', cmap='gray')
        plt.title(f"Sinogram at Z={z_index}, Energy Bin {energy_bin_index}")
        plt.xlabel("Detector Y Pixel")
        plt.ylabel("Projection Index")
        plt.colorbar(label="Photon Counts")
        plt.tight_layout()
        plt.savefig(f'sinograms/sinogram_counts_{name}_{detector_shape[0]}_{detector_shape[1]}_0.1scaling.png')
        plt.show()
    else:
        plt.figure(figsize=(6, 4))
        plt.imshow(attenuation, aspect='auto', cmap='gray')
        plt.title(f"Sinogram at Z={z_index}, Energy Bin {energy_bin_index}")
        plt.xlabel("Detector Y Pixel")
        plt.ylabel("Projection Index")
        plt.colorbar(label="Attenuation")
        plt.tight_layout()
        plt.savefig(f'sinograms/sinogram_attenuation_{name}_{detector_shape[0]}_{detector_shape[1]}_0.1scaling.png')
        plt.show()


if __name__ == "__main__":
    objectSize = 32
    pixelPitch = 1
    nPixelsY = 64
    nPixelsZ = 44
    d_detector = 900
    dSource = 2300
    n_proj = 100
    projAngles = np.arange(0, n_proj) * (np.pi / n_proj)  # -np.pi/2
    astra = 1  # 0: conebeamtomo3, 1: astra matrix, 2: astra projector

    if astra == 2:
        name = "astra"
    elif astra == 1:
        name = "astramatrix"
    else:
        name = "original"

    # project matrices
    y, S, M, A, x = simulatepreps(objectSize, nPixelsY, nPixelsZ, pixelPitch, d_detector, dSource, projAngles, astra, scale=0.1, idx_phantom=0)
    print("total data size: ", y.size)
    showSinogram(y, name,type =1, energy_bin_index=0, z_index=22, nProj=100, detector_shape=(nPixelsY, nPixelsZ), )