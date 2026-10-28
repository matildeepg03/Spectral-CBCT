import numpy as np
from astra_sinogram import astra_sino
from load_gate import load_gatesino
from phantom_function_more_materials import phantom_astra
from conebeamtomo3 import conebeamtomo3
from noisyforwardmodel import noisyforwardmodel
from astra_file import get_astra_projection_matrix, matrix_generation
import xraydb

def simulatepreps(object_size, object_depth, n_pixels_y, n_pixels_z, pixel_pitch, d_detector, dSource, projAngles, astra, scale, n_proj):
    """
    Simulate the photon counts with a phantom.

    Parameters:
        object_size (int): Length of one side of a cubed object.
        n_pixels_y (int): Number of pixels in Y direction.
        n_pixels_z (int): Number of pixels in Z direction.
        pixel_pitch (float): Distance between the centers of the pixels in mm.

    Returns:
        y (ndarray): Set of preps (nBins x nPixels).
        S (ndarray): Bin photon count (nBins x nEnergies).
        M (ndarray): Material Attenuation for each energy (nEnergies x nMats).
        A (ndarray): Forward Projection matrix (nPixels x (object_size^3)).
    """
    """
    ######### MATERIAL ATTENUATION MATRIX & ENERGY SPECTRUM WITH MONOENERGETIC SOURCE ###########

    # Keep the monoenergetic spectrum compact so the forward model stays small in memory.
    mono_energies = np.array([35000, 75000], dtype=float)

    # Define the compositions of water and bone
    water_composition = {'O': 0.888102, 'H': 0.111898}
    bone_composition = {'O': 0.435, 'C': 0.155, 'H': 0.034, 'N': 0.042, 'Na': 0.001, 'Mg': 0.002, 'P': 0.103,
                        'S': 0.003, 'Ca': 0.225}

    mu_rho_water = np.zeros_like(mono_energies, dtype=float)
    mu_rho_bone = np.zeros_like(mono_energies, dtype=float)

    for element, fraction in water_composition.items():
        mu_rho_water += fraction * xraydb.mu_elam(element, mono_energies)

    for element, fraction in bone_composition.items():
        mu_rho_bone += fraction * xraydb.mu_elam(element, mono_energies)

    mu_water = mu_rho_water * 1
    mu_bone = mu_rho_bone * 1.92

    # Create the material attenuation matrix M
    M = np.column_stack([mu_bone, mu_water])
    # I0 calculation (particles per pixel)
    I_source = 1e7
    pixel_area = pixel_pitch ** 2
    SDD = dSource + d_detector
    half_angle = np.deg2rad(0.69)
    term2 = pixel_area / (4 * np.pi * SDD ** 2)
    term3 = 1 / (np.sin(half_angle / 2) ** 2)
    I_total = I_source * term2 * term3

    probabilities = np.array([0.5, 0.5], dtype=float)  # Probability for each mid bin energy to occur at the detector.
    S = np.diag(probabilities) * I_total # Create the bin photon count matrix S

    """
    ######### MATERIAL ATTENUATION MATRIX & ENERGY SPECTRUM WITH POLIENERGETIC SOURCE ###########

    # Mid-bin energy spectrum from 1.5 keV to 99.5 keV with 1 keV increments
    mid_bin_energy_spectrum = np.arange(1500, 99501, 1000)

    # Define the compositions of water and bone
    water_composition = {'O': 0.888102, 'H': 0.111898}
    bone_composition = {'O': 0.435, 'C': 0.155, 'H': 0.034, 'N': 0.042, 'Na': 0.001, 'Mg': 0.002, 'P': 0.103,
                        'S': 0.003, 'Ca': 0.225}

    mu_rho_water = np.zeros_like(mid_bin_energy_spectrum, dtype=float)
    mu_rho_bone = np.zeros_like(mid_bin_energy_spectrum, dtype=float)

    for element, fraction in water_composition.items():
        mu_rho_water += fraction * xraydb.mu_elam(element, mid_bin_energy_spectrum)

    for element, fraction in bone_composition.items():
        mu_rho_bone += fraction * xraydb.mu_elam(element, mid_bin_energy_spectrum)

    mu_water = mu_rho_water * 1
    mu_bone = mu_rho_bone * 1.92

    M = np.column_stack([mu_bone, mu_water])

    # I0 calculation (particles per pixel)
    I_source = 1e7
    pixel_area = pixel_pitch ** 2
    SDD = dSource + d_detector
    half_angle = np.deg2rad(1.38)
    term2 = pixel_area / (4 * np.pi * SDD ** 2)
    term3 = 1 / (np.sin(half_angle / 2) ** 2)
    I_total = I_source * term2 * term3
    print(f"Total number of photons per pixel: {I_total:.2e}")

    # Probability for each mid bin energy to occur at the detector (obtained with speakpy)
    probabilities = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0003, 0.0003, 0.0007, 0.0014, 0.0024, 0.0036, 0.0051, 0.0067, 0.0085, 0.0102, 0.0119, 0.0136, 0.0151, 0.0164, 0.0177, 0.0187, 0.0196, 0.0204, 0.021, 0.0214, 0.0217, 0.0219, 0.022, 0.022, 0.0219, 0.0218, 0.0216, 0.0213, 0.021, 0.0207, 0.0203, 0.0199, 0.0195, 0.0191, 0.0187, 0.0182, 0.0178, 0.0173, 0.0168, 0.0164, 0.0159, 0.0155, 0.015, 0.0146, 0.0141, 0.0137, 0.0257, 0.0129, 0.0346, 0.012, 0.0116, 0.0112, 0.0109, 0.0105, 0.0101, 0.0124, 0.0147, 0.0091, 0.0101, 0.007, 0.0068, 0.0066, 0.0063, 0.0061, 0.0059, 0.0057, 0.0054, 0.0052, 0.005, 0.0047, 0.0045, 0.0043, 0.004, 0.0038, 0.0036, 0.0034, 0.0031, 0.0029, 0.0027, 0.0024, 0.0022, 0.0019, 0.0017, 0.0014, 0.0012, 0.001, 0.0007, 0.0004, 0.0005], dtype=float)
    
    # Create the bin photon count matrix S
    S_bin1 = [0] * len(probabilities)
    S_bin1[19:49] = probabilities[19:49] * I_total

    S_bin2 = [0] * len(probabilities)
    S_bin2[50:98] = probabilities[50:98] * I_total

    S = np.column_stack([S_bin1, S_bin2])
    S=S.T
 
    energy_mask = np.any(S != 0, axis=0).astype(int)  # these are energies that actually participate
    M *= energy_mask[:, None] # this way we do not calculate exp(-M*A*x) for the energies that don't particpate in the end. because that gives divergences

    x, bone, water = phantom_astra(object_size, object_depth)

    if astra==2:
        A_full, _, _, _ = get_astra_projection_matrix(object_size, pixel_pitch, n_pixels_y, n_pixels_z, projAngles,
                                                      dSource,
                                                      d_detector, scale)
        A = matrix_generation(A_full, object_size)

    elif astra == 1 :
        A, _, _, _ = get_astra_projection_matrix(object_size, pixel_pitch, n_pixels_y, n_pixels_z, projAngles, dSource,
                                                 d_detector, scale)

    elif astra == 0:
        A = conebeamtomo3(object_size, np.rad2deg(projAngles), pixel_pitch, n_pixels_y, n_pixels_z)


    # ASTRA forward model
    #y= astra_sino(object_size, n_pixels_y, n_pixels_z, projAngles, scale, dSource, d_detector, x, M, S) # astra data
    
    # ASTRA w/noise forward model
    #y = noisyforwardmodel(x, A, S, M, astra)
    
    # GATE projections dataset
    low = np.load("low_64correct.npy")
    high = np.load("high_64correct.npy")
    y = load_gatesino(low, high)
    np.save("photoncounts_GATE.npy", y)
    
    return y, S, M, A, x, bone, water

