import numpy as np
import matplotlib.pyplot as plt

def unload_astra(y, shape):
    """
    Reconstruct two ASTRA sinograms from flattened representation.

    Parameters
    - y: ndarray with shape (2, nPixelsz * nProj * nPixelsY)
    - shape: tuple (nPixelsZ, nProj, nPixelsY)

    Returns
    - list [bin1, bin2] with each sinogram shaped (nPixelsZ, nProj, nPixelsY)
    """

    nz, nProj, ny = shape

    nPixels = nz * nProj * ny

    if y.shape[1] != nPixels:
        raise ValueError(
            f"Expected flattened size {nPixels}, got {y.shape[1]}"
        )

    sinograms = []

    for i in range(y.shape[0]):  # for each bin
        flat = y[i]

        
        s = flat.reshape(nz, nProj, ny) # for astra
        #s = flat.reshape(nProj, ny, nz).transpose(2, 0, 1)  # for gate

        sinograms.append(s)

    return sinograms


y = np.load("photoncounts_GATE.npy")
print("Shape of y:", y.shape)
sinogram1, sinogram2 = unload_astra(y, (44, 360, 64))
print("min and max of sinogram1:", sinogram1.min(), sinogram1.max())
 

nz = 22
nproj = 50
ny = 32

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

print(sinogram1.shape)

# Axial
im0 = axes[0].imshow(sinogram1[nz, :, :], cmap='gray', vmin=0, vmax=sinogram1.max())
axes[0].set_title(f'Fixed z = {nz}')
axes[0].set_xlabel('Pixel y')
axes[0].set_ylabel('Projection Angle θ')

# Sagittal
coronal = sinogram1[:, nproj, :].T
im1 = axes[1].imshow(coronal, cmap='gray', vmin=0, vmax=sinogram1.max())
axes[1].set_title(f'Fixed Angle θ = {nproj}')
axes[1].set_xlabel('Pixel x')
axes[1].set_ylabel('Pixel y')

# Coronal
im2 = axes[2].imshow(sinogram1[:, :, ny].T, cmap='gray', vmin=0, vmax=sinogram1.max())
axes[2].set_title(f'Fixed y = {ny}')
axes[2].set_xlabel('Pixel x')
axes[2].set_ylabel('Projection Angle θ')



# Add colorbars
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
