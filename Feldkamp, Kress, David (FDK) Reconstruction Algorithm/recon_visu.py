import numpy as np
import matplotlib.pyplot as plt

# Load your 3D array
volume = np.load("recon_high_stack.npy")
#normalized_volume = volume / np.max(volume)
print("Shape:", volume.shape)
print("Max value in volume:", volume.max())
print("Min value in volume:", volume.min())

# Take center slices
z, y, x = (np.array(volume.shape) / 2).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
recon_max_low = 0.03074712633834195
recon_max_high = 0.018791504993379775

# Axial
im1 = axes[0].imshow(volume[z, :, :], cmap='gray', vmin=0, vmax=recon_max_high)
axes[0].set_title(f'Axial (z={z})')
axes[0].set_xlabel('x pixel')
axes[0].set_ylabel('y pixel')
# Coronal
im2 = axes[1].imshow(volume[:, y, :], cmap='gray', vmin=0, vmax=recon_max_high)
axes[1].set_title(f'Coronal (y={y})')
axes[1].set_xlabel('x pixel')
axes[1].set_ylabel('z pixel')
# Sagittal
im3 =  axes[2].imshow(volume[:, :, x], cmap='gray', vmin=0, vmax=recon_max_high)
axes[2].set_title(f'Sagittal (x={x})')
axes[2].set_xlabel('y pixel')
axes[2].set_ylabel('z pixel ')
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()


