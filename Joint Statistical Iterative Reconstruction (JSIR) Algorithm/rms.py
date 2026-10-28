
import numpy as np
import matplotlib.pyplot as plt

def calculate_rms_error(iterates, phantom, objectSize):
    """
        Calculates the RMS error between the phantom and the reconstruction.
    """
    _, nMats, nIterates = iterates.shape
    mid_idx = (objectSize // 2) - 1

    matres = iterates.reshape((objectSize, objectSize, objectSize, nMats, nIterates), order='F')

    matres1 = matres[:, :, mid_idx, 0, -1]
    matres2 = matres[:, :, mid_idx, 1, -1]

    # 2. Extract flat phantom materials
    bone_flat = phantom[:, 0]
    water_flat = phantom[:, 1]

    bone_3d = bone_flat.reshape((objectSize, objectSize, objectSize), order='F')
    water_3d = water_flat.reshape((objectSize, objectSize, objectSize), order='F')


    bone_slice = bone_3d[:, :, mid_idx]
    water_slice = water_3d[:, :, mid_idx]

    squared_error_bone = (bone_slice - matres1) ** 2
    squared_error_water = (water_slice - matres2) ** 2

    mse_bone = np.mean(squared_error_bone)
    mse_water = np.mean(squared_error_water)

    tot = np.append(squared_error_bone, squared_error_water)
    mse_total = np.mean(tot)

    rmse_bone = np.sqrt(mse_bone)
    rmse_water = np.sqrt(mse_water)
    rmse_total = np.sqrt(mse_total)


    return rmse_bone, rmse_water, rmse_total

def iteration_needed(iterates, threshold=1e-2):
    """"
    Calculates the number of iterations needed for the reconstruction to reach a certain treshold of change"""
    _, _, nIterates = iterates.shape

    for i in range(1, nIterates):
        prev_step = iterates[:, :, i - 1]
        curr_step = iterates[:, :, i]


        diff_norm = np.linalg.norm(curr_step - prev_step)
        prev_norm = np.linalg.norm(prev_step)

        if prev_norm == 0:
            continue

        relative_change = diff_norm / prev_norm

        if relative_change < threshold:
            #print(f"Converged to relative change < {threshold} at iteration: {i}")
            return i

    #print(f"Did not converge to {threshold} within {nIterates} iterations.")
    return nIterates
