""" # To visualize ASTRA Projections
import numpy as np
import matplotlib.pyplot as plt

# Load your 3D array
volume = np.load("forw_0.npy")
print(volume.shape)
print("Max value in volume:", volume.max())
print("Min value in volume:", volume.min())
#normalized_volume = volume / np.max(volume)
# Take center slices
#x, z, y = (np.array(volume.shape) / 2).astype(int)
x = 32
y =32
z= 32
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
proj_max_low = 1.0998968
proj_max_high = 0.6722155
print(volume.shape)

# Axial
im0 = axes[0].imshow(volume[:, z, :].transpose(1, 0), cmap='gray', vmin=0, vmax=volume.max())
axes[0].set_title(f'Fixed Angle θ = {z}')
axes[0].set_xlabel('Pixel x')
axes[0].set_ylabel('Pixel y')

# Sagittal
coronal = volume[:, :, y].T
im1 = axes[1].imshow(coronal, cmap='gray', vmin=0, vmax=volume.max())
axes[1].set_title(f'Fixed y = {y}')
axes[1].set_xlabel('Pixel x')
axes[1].set_ylabel('θ')

# Coronal
im2 = axes[2].imshow(volume[x, :, :], cmap='gray', vmin=0, vmax=volume.max())
axes[2].set_title(f'Fixed x = {x}')
axes[2].set_xlabel('Pixel y')
axes[2].set_ylabel('θ')



# Add colorbars
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
 """

# To visualize GATE Projections 
import numpy as np
import matplotlib.pyplot as plt

# Load your 3D array
volume = np.load("correct_low.npy")
volume = np.transpose(volume, (2, 1, 0))  # Transpose to match the expected shape
print("Shape:", volume.shape)

# Take center slices
#z, y, x = (np.array(volume.shape) / 2).astype(int)
nz = 22
nproj = 0
ny = 32

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
proj_max_low = 1.0998968
proj_max_high = 0.6722155

# Axial
im0 = axes[0].imshow(volume[nz, :, :], cmap='gray', vmin=0, vmax=volume.max())
axes[0].set_title(f'Fixed Angle θ = {nz}')
axes[0].set_xlabel('Pixel x')
axes[0].set_ylabel('Pixel y')

# Coronal
im1 = axes[1].imshow(volume[:, nproj, :], cmap='gray', vmin=0, vmax=volume.max())
axes[1].set_title(f'Fixed y = {nproj}')
axes[1].set_xlabel('Pixel x')
axes[1].set_ylabel('θ')

# Sagittal
im2 = axes[2].imshow(volume[:, :, ny].T, cmap='gray', vmin=0, vmax=volume.max())
axes[2].set_title(f'Fixed x = {ny}')
axes[2].set_xlabel('Pixel y')
axes[2].set_ylabel('θ')

# Add colorbars
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
