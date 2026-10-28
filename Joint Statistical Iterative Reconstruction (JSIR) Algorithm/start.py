import time
from simulatepreps_phantom import simulatepreps
from Mechlem2017 import mechlem2017
from playiterates import playiterates
from phantom import phantom_print
import numpy as np
from VMIs import calculate_crc, mask_roi_water, mask_roi_bone, compute_vmi, roi_statistics
from spr import compare_to_ground_truth, spr_show
from profile import plot_spr_profile, plot_water_profile

s = time.time()
# Reconstruction parameters
objectSize = 64
objectDepth = 6
nIter = 20
nSubset = 4
print(f"objectSize: {objectSize}, objectDepth: {objectDepth}, nIter: {nIter}, nSubset: {nSubset}")
# Simulation parameters
nPixelsY = 128
nPixelsZ = 88
pixelPitch = 1
d_detector = 900
dSource = 2300
n_proj = 360
print(f"nPixelsY: {nPixelsY}, nPixelsZ: {nPixelsZ}, n_proj: {n_proj}")
projAngles = np.arange(0, n_proj) * (np.pi / n_proj)
nPixelsPerProj = nPixelsZ * nPixelsY

#VMI parameters
E1 = 35000
E2 = 75000

astra = 1 #0: conebeamtomo3, 1: astra matrix, 2: astra projector
save = True
scale = 0.1 #do not touch, this is to adjust the astra matrix and projector to work with attenuation factors etc

# Regularization parameters
#lambda_ = np.array([0.0001, 0.0001]) # ASTRA REGULARIZATION
lambda_ = np.array([0.1, 0.3]) # GATE REGULARIZATION
print(f"lambda_: {lambda_}")
delta_huber = np.array([0.05, 0.07])
print(f"delta_huber: {delta_huber}")

# Save directory for results
run_name = "all_no_noise_nProj_360" #change name for file based on what you are running eg. original matrix vs astra projector
file_name = f"{run_name}_{objectSize}_{nIter}_{nSubset}_{scale}_{lambda_}"
title = f"nIter = {nIter}, nSubset = {nSubset}, scale = {scale}, lambda_ = {lambda_}"

# Initialize arrays to store costs and times
totalCosts = np.zeros((3, nIter))
dataAttCosts = np.zeros((3, nIter))
regCosts = np.zeros((3, nIter))
times_list = np.zeros((3, nIter))


for i in [astra]: #remember to change the plot size in phantom_print and the ordering in play iterates when running only some
    start_time = time.time()
    astra = i

# Simulate preparation
    y, S, M, A, x, bone, water = simulatepreps(objectSize, objectDepth, nPixelsY, nPixelsZ, pixelPitch, d_detector, dSource, projAngles, astra, scale, n_proj)

    if astra == 2:
        name = "astra"
        title_name = "Astra"

    elif astra == 1:
        name = "astramatrix"
        title_name = "Astra Matrix"

    else:
        name = "original"
        title_name = "conebeamtomo3.py"

    # Perform reconstruction
    Mechlem2017_iterates, totalCost, dataAttCost, regCost, times = mechlem2017(objectSize, y, S, M, A, nPixelsPerProj, nIter, nSubset, lambda_, delta_huber, projAngles, pixelPitch, nPixelsY, nPixelsZ, dSource, d_detector, astra, scale, nesterov = True)
    _, nMats, nIterates = Mechlem2017_iterates.shape

    # Display results
    fig, axes = phantom_print(x, objectSize, title)
    matres1, matres2 = playiterates(Mechlem2017_iterates, [0, 0, 0], [1.5, 1.5, 0.5], objectSize,  title_name, nIter, nSubset, save, astra, file_name, fig, axes)

    VMI_E1, VMI_E2, mu_water, mu_bone = compute_vmi(matres1, matres2, E1, E2)

    mask_water = mask_roi_water(objectSize)
    statistics_water_E1 = roi_statistics(VMI_E1, mask_water)
    statistics_water_E2 = roi_statistics(VMI_E2, mask_water)

    mask_bone = mask_roi_bone(objectSize)
    statistics_bone_E1 = roi_statistics(VMI_E1, mask_bone)
    statistics_bone_E2 = roi_statistics(VMI_E2, mask_bone)

    plot_water_profile(matres1, matres2, (0, objectSize-1), (objectSize-1, 0), pixel_size=pixelPitch)

    slice_idx = (objectSize // 2)-1
    bone_true = bone[slice_idx, :, :]
    water_true = water[slice_idx, :, :]
    vmi_true1 = bone_true * mu_bone[0] + water_true * mu_water[0]
    vmi_true2 = bone_true * mu_bone[1] + water_true * mu_water[1]

    C_recon1, crc1 = calculate_crc(VMI_E1, vmi_true1, mask_bone, mask_water)

    C_recon2, crc2 = calculate_crc(VMI_E2, vmi_true2, mask_bone, mask_water)

    print("Statistics VMI E1:")
    print(f"Water: {statistics_water_E1}")
    print(f"Bone: {statistics_bone_E1}")
    print(f"CRC: {crc1}")
    print(f"C: {C_recon1}")
    print("Statistics VMI E2:")
    print(f"Water: {statistics_water_E2}")
    print(f"Bone: {statistics_bone_E2}")
    print(f"CRC: {crc2}")
    print(f"C: {C_recon2}")

    SPR_gt, SPR_rc = spr_show(water_true, bone_true, matres2, matres1)
    plot_spr_profile(
        SPR_gt,
        SPR_rc,
        (0, objectSize - 1),
        (objectSize - 1, 0),
        pixel_size=pixelPitch
    )

    print("Statistics SPR:")
    stat = compare_to_ground_truth(SPR_rc, SPR_gt)
    print(stat)

    e = time.time()
    #print(f"Total time for astra old = {astra}: {e - start_time} seconds")