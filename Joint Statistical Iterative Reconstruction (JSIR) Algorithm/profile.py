from skimage.measure import profile_line
import matplotlib.pyplot as plt
import numpy as np


def plot_water_profile(bone_map, water_map, start_point, end_point, pixel_size=1.0):

    """
    Extract and plot a water material line profile.

    Parameters
    ----------
    bone_map : 2D numpy array
        Bone material decomposition map.
    water_map : 2D numpy array
        Water material decomposition map.
    start_point : tuple
        Starting point (row, col).
    end_point : tuple
        Ending point (row, col).
    pixel_size : float
        Pixel size in mm.
    """

    water_profile = profile_line(
        water_map,
        start_point,
        end_point,
        mode='constant',
        cval=np.nan
    )

    bone_profile = profile_line(
        bone_map,
        start_point,
        end_point,
        mode='constant',
        cval=np.nan
    )

    distance = np.arange(len(water_profile)) * pixel_size

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(distance, water_profile, label='Water', color = 'blue')
    ax.plot(distance, bone_profile, label='Bone', color = 'yellow')

    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Coefficient")
    ax.set_title("Material map profile")
    ax.grid(True)
    ax.legend()

    fig.savefig(
        '/scratch/matildegoncalv/material_map_profile.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    return distance


def plot_spr_profile(spr_ground_truth, spr_reconstruction, start_point, end_point, pixel_size=1.0):
    """Plot SPR ground truth and reconstruction along the same image line."""
    ground_truth_profile = profile_line(
        spr_ground_truth,
        start_point,
        end_point,
        mode='constant',
        cval=np.nan
    )
    reconstruction_profile = profile_line(
        spr_reconstruction,
        start_point,
        end_point,
        mode='constant',
        cval=np.nan
    )

    distance = np.arange(len(ground_truth_profile)) * pixel_size

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(distance, ground_truth_profile, label='Ground Truth', color='black')
    ax.plot(distance, reconstruction_profile, label='Reconstruction', color='red')
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("SPR")
    ax.set_title("SPR Profile")
    ax.grid(True)
    ax.legend()

    fig.savefig(
        '/scratch/matildegoncalv/SPR_profile.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    return distance