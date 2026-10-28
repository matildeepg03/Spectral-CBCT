""" # To visualize ASTRA Projections
import numpy as np
import matplotlib.pyplot as plt

# Load your 3D array
volume = np.load("projections_high.npy")
#normalized_volume = volume / np.max(volume)
# Take center slices
#x, z, y = (np.array(volume.shape) / 2).astype(int)
x = 128
y =128
z=90
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
proj_max_low = 1.0998968
proj_max_high = 0.6722155
print(volume.max())

# Axial
im0 = axes[0].imshow(volume[:, z, :], cmap='gray', vmin=0, vmax=proj_max_high)
axes[0].set_title(f'Fixed Angle θ = {z}')
axes[0].set_xlabel('Pixel x')
axes[0].set_ylabel('Pixel y')

# Coronal
im1 = axes[1].imshow(volume[x, :, :], cmap='gray', vmin=0, vmax=proj_max_high)
axes[1].set_title(f'Fixed x = {x}')
axes[1].set_xlabel('Pixel y')
axes[1].set_ylabel('θ')

# Sagittal
coronal = volume[:, :, y].T
im2 = axes[2].imshow(coronal, cmap='gray', vmin=0, vmax=proj_max_high)
axes[2].set_title(f'Fixed y = {y}')
axes[2].set_xlabel('Pixel x')
axes[2].set_ylabel('θ')

# Add colorbars
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show() """


# To visualize GATE Projections 
import numpy as np
import matplotlib.pyplot as plt

# Load your 3D array
volume = np.load("atten_low2.npy")
print("Max value in volume:", volume.max())
print("Min value in volume:", volume.min())
print("Shape:", volume.shape)

# Take center slices
#z, y, x = (np.array(volume.shape) / 2).astype(int)
z = 90
y = 128
x = 128

print(z, y, x)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
proj_max_low = 1.0998968
proj_max_high = 0.6722155
#print(volume[180, 130:132, 125:132])

# Axial
im0 = axes[0].imshow(volume[z, :, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
axes[0].set_title(f'Fixed Angle θ = {z}')
axes[0].set_xlabel('Pixel x')
axes[0].set_ylabel('Pixel y')

# Coronal
im1 = axes[1].imshow(volume[:, y, :], cmap='gray', vmin=volume.min(), vmax=volume.max())
axes[1].set_title(f'Fixed y = {y}')
axes[1].set_xlabel('Pixel x')
axes[1].set_ylabel('θ')

# Sagittal
im2 = axes[2].imshow(volume[:, :, x], cmap='gray', vmin=volume.min(), vmax=volume.max())
axes[2].set_title(f'Fixed x = {x}')
axes[2].set_xlabel('Pixel y')
axes[2].set_ylabel('θ')

# Add colorbars
fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
