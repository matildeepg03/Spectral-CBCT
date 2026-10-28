import astra
import numpy as np

# Load projection data
projections = np.load("correct64_low.npy")
print("projections shape:", projections.shape)

projections = np.transpose(projections, (2, 1, 0)) # only with gate data
print("projections shape after transpose:", projections.shape)
#print("should be 256, 360, 256")

# Parameters
nz, num_angles, ny = projections.shape
n_proj = 360
N=64
angles = np.arange(0, n_proj) * (np.pi / n_proj)
voxel_size = 1 # cm/pixel
src_obj_dist = 2300
obj_det_dist = 900

# Define volume geometry
lim = total_width_cm = (N * voxel_size) / 2
vol_geom = astra.create_vol_geom(N, N, N, -lim, lim, -lim, lim, -lim, lim)

# Define projection geometry
det_spacing = voxel_size
proj_geom = astra.create_proj_geom('cone', det_spacing, det_spacing, nz, ny, angles, src_obj_dist, obj_det_dist)

# Create ASTRA 3D projection data object
sinogram_id = astra.data3d.create('-proj3d', proj_geom, projections)

# Create reconstruction data object
recon_id = astra.data3d.create('-vol', vol_geom)

# FDK reconstruction
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = recon_id
algorithm_id = astra.algorithm.create(cfg)
astra.algorithm.run(algorithm_id)

# Get the reconstructed volume
reconstruction = astra.data3d.get(recon_id)

print("min, max reconstruction:", reconstruction.min(), reconstruction.max())
#print("reconstruction value at [128, 128, 128]:", reconstruction[128, 128, 128])
#print("reconstruction value at [1, 1, 1]:", reconstruction[1, 1, 1])

# Save full 3D reconstruction
np.save("recon_low64.npy", reconstruction)

def mask_roi_water(objectsize):
    radius = (objectsize * 0.3) / 2
    center = objectsize / 2
    size = objectsize

    Y, X = np.ogrid[:size, :size]

    water_mask = (
        (X - center)**2 +
        (Y - center)**2
        <= radius**2
    )

    return water_mask


def mask_roi_bone(objectsize):
    num_rods = 8
    center = objectsize / 2
    size = objectsize

    radius_outer = objectsize * 0.4
    radius_small_circles = objectsize * 0.03

    Y, X = np.ogrid[:size, :size]

    mask = np.zeros((size, size), dtype=bool)

    for i in range(num_rods):
        angle = i * 2 * np.pi / num_rods

        cx = center + int(
            (radius_outer / 2) * np.cos(angle)
        )

        cy = center + int(
            (radius_outer / 2) * np.sin(angle)
        )

        rod_mask = (
            (X - cx)**2 +
            (Y - cy)**2
            <= radius_small_circles**2
        )

        mask |= rod_mask

    return mask

def roi_statistics(image, mask):

    values = image[mask]

    if values.size < 2:
        raise ValueError(
            "The ROI must contain at least two pixels."
        )

    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4),
        "variance": round(float(np.var(values, ddof=1)), 4),
        "minimum": round(float(np.min(values)), 4),
        "maximum": round(float(np.max(values)), 4),
    }


def calculate_image_metrics(image, bone_mask, water_mask):

    bone_stats = roi_statistics(image, bone_mask)
    water_stats = roi_statistics(image, water_mask)

    mean_bone = np.mean(image[bone_mask])
    mean_water = np.mean(image[water_mask])

    C = (mean_bone - mean_water) / mean_water

    return {
        "bone": bone_stats,
        "water": water_stats,
        "C": C
    }


# Central FDK slice
slice_idx = (N // 2) - 1
fdk_slice = reconstruction[slice_idx, :, :]

# Create ROIs
mask_water = mask_roi_water(N)
mask_bone = mask_roi_bone(N)

# Calculate metrics
fdk_E1 = calculate_image_metrics(
    fdk_slice,
    mask_bone,
    mask_water
)

print("\nFDK Low-Energy Metrics:")
print("Water:", fdk_E1["water"])
print("Bone:", fdk_E1["bone"])
print("Contrast:", fdk_E1["C"])