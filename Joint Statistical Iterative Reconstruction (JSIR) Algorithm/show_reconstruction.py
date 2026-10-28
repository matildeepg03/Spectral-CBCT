import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from phantom import phantom_square
# Object Size a square
objectSize = 32
#
# Number of iterations
nIter = 100


# Number of subsets
nSubset = 4

# Number of pixels on the detector plate.

nPixelsY = 64
nPixelsZ = 44

# Pixel pitch
pixelPitch = 1

d_detector = 900
dSource = 2300

n_proj = 150
projAngles = np.arange(0, n_proj) * (np.pi / n_proj)

# The total number of pixels on the detector plate
nPixelsPerProj = nPixelsZ * nPixelsY


astra = 0 #0: conebeamtomo3, 1: astra matrix, 2: astra projector
save = True #save plots
scale = 0.1 #do not touch, this is to adjust the astra matrix and projector to work with attenuation factors etc
idx_phantom =1 #select which phantom geometry you want
noise = True

# # Set regularization parameters
lambda_ = np.array([10, 5]) # From [100, 50]
delta_huber = np.array([0.01, 0.05]) #[0.01, 0.05]

current_dir = Path(__file__).parent

run_name = "all_no_noise_nProj_150" #change name for file based on what you are running eg. original matrix vs astra projector
file_name = f"{run_name}_{objectSize}_phantom{idx_phantom}_{nIter}_{nSubset}_{scale}_{lambda_}"
title = f"nIter = {nIter}, nSubset = {nSubset}, scale = {scale}, lambda_ = {lambda_}"
x = phantom_square(objectSize, idx_phantom)

bone = x[:, 0].reshape((objectSize, objectSize, objectSize), order='F')
water = x[:, 1].reshape((objectSize, objectSize, objectSize), order='F')

bone /= 1.92
water /= 1
mid = objectSize // 2
# Display ground truth images
fig, axes = plt.subplots(3, 2, figsize=(8, 18))
# plt.figure(figsize=(8, 10))
fig.suptitle(f"nIter = {nIter}, nSubset: {nSubset}, detector: {nPixelsY}x{nPixelsZ}, objectSize = {objectSize}, ")
Im2 = axes[0, 1].imshow(bone[:, :, mid], cmap='gray', vmin=0, vmax=1, interpolation='none')
axes[0, 1].set_title(f'Bone Phantom')
fig.colorbar(Im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

Im1 = axes[0, 0].imshow(water[:, :, mid], cmap='gray', vmin=0, vmax=1, interpolation='none')
axes[0, 0].set_title(f'Water Phantom')
fig.colorbar(Im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
# plt.imshow(bone[:, :, mid], cmap='gray', vmin=0, vmax=1, interpolation='none')
# plt.title('bone')
# plt.colorbar()
# plt.show()
#
# plt.imshow(water[:, :, mid], cmap='gray', vmin=0, vmax=1, interpolation='none')
# plt.title('water')
# plt.colorbar()
# plt.show()
if noise:
    directory =  f"npy_files_noise_idx{idx_phantom}"
else:
    directory =  f"npy_files_idx{idx_phantom}"

iterates_path_0 = current_dir.parent / "code_simple_water_bone" / directory / f'Mechlem2017_iterates_0_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
iterates_0 = np.load(str(iterates_path_0))
_, nMats, nIterates = iterates_0.shape

matres_0 = iterates_0.reshape((objectSize, objectSize, objectSize, nMats, nIterates))  # C-order
matres1_0 = matres_0[mid, :, :, 0, -1]  # bone, slab axis
matres2_0 = matres_0[mid, :, :, 1, -1]  # water



Im2 = axes[1, 1].imshow(matres1_0, cmap='gray', vmin=0, vmax=1, interpolation='none')
axes[1, 1].set_title(f'Bone Reconstruction conebeamtomo3.py')
fig.colorbar(Im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

Im1 = axes[1, 0].imshow(matres2_0, cmap='gray', vmin=0, vmax=1, interpolation='none')
axes[1, 0].set_title(f'Water  Reconstruction conebeamtomo3.py')
fig.colorbar(Im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

iterates_path_1 = current_dir.parent / "code_simple_water_bone" / directory / f'Mechlem2017_iterates_1_{nIter}_{nSubset}_{objectSize}_{nPixelsY}_{nPixelsZ}_{lambda_}.npy'
iterates_1 = np.load(str(iterates_path_1))
_, nMats, nIterates = iterates_1.shape

matres_1 = iterates_1.reshape((objectSize, objectSize, objectSize, nMats, nIterates))  # C-order
matres1_1 = matres_1[mid, :, :, 0, -1]  # bone, slab axis
matres2_1 = matres_1[mid, :, :, 1, -1]  # water



Im2 = axes[2, 1].imshow(matres1_1, cmap='gray', vmin=0, vmax=1, interpolation='none')
axes[2, 1].set_title(f'Bone Reconstruction conebeamtomo3.py')
fig.colorbar(Im2, ax=axes[2, 1], fraction=0.046, pad=0.04)

Im1 = axes[2, 0].imshow(matres2_1, cmap='gray', vmin=0, vmax=1, interpolation='none')
axes[2, 0].set_title(f'Water  Reconstruction conebeamtomo3.py')
fig.colorbar(Im1, ax=axes[2, 0], fraction=0.046, pad=0.04)
# # Display images
# plt.imshow(matres1, cmap='gray', vmin=0, vmax=1, interpolation='none')
# plt.title('Maybe bone')
# plt.colorbar()
# plt.savefig('Python_reconstruction_32x32_bone.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white')
# plt.show()
#
# plt.imshow(matres2, cmap='gray', vmin=0, vmax=1, interpolation='none')
# plt.title('Maybe water')
# plt.colorbar()
# plt.savefig('Python_reconstruction_32x32_water.png', bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='white')
# plt.show()

plt.show()


