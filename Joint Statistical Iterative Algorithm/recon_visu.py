import numpy as np
import matplotlib.pyplot as plt

# Load your 3D array
volume = np.load("recon_low2.npy")
#normalized_volume = volume / np.max(volume)
#volume = np.transpose(volume, (0, 2, 1))
print("Shape:", volume.shape)
print("Max value in volume:", volume.max())
print("Min value in volume:", volume.min())
#print("Value at [128, 128, 128]:", volume[128, 128, 128])
# Take center slices
z, y, x = (np.array(volume.shape) / 2).astype(int)

recon_max_low = 0.03074712633834195
recon_max_high = 0.018791504993379775


fig, ax = plt.subplots(figsize=(4, 4))

im = ax.imshow(volume[x, :, :], 
               cmap='gray',
               vmin=0,
               vmax=volume.max())

ax.set_title("Energy Bin 1\n(20-50 keV)", fontsize=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\mu$ (cm$^{-1}$)')

ax.tick_params(labelsize=8)

plt.tight_layout(pad=0.2)
plt.show()


