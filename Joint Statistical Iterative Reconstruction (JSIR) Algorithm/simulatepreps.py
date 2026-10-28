import numpy as np
import scipy.io
from scipy.io import savemat

from conebeamtomo3 import conebeamtomo3
from noisyforwardmodel import noisyforwardmodel


def simulatepreps(object_size, n_pixels_y, n_pixels_z, pixel_pitch):
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

    # Load MATLAB .mat files containing the required data
    incident_spectrum = scipy.io.loadmat('/Users/MaxineVanHoutven1/PycharmProjects/maxine-project/Matlab_files/incidentSpectrum.mat')['incidentSpectrum']
    material_attenuations = scipy.io.loadmat('/Users/MaxineVanHoutven1/PycharmProjects/maxine-project/Matlab_files/materialAttenuationBoneWater.mat')['materialAttenuations']
    detector_response = scipy.io.loadmat('/Users/MaxineVanHoutven1/PycharmProjects/maxine-project/Matlab_files/detectorResponse.mat')['detectorResponse']

    # Material attenuations (nEnergies x nMats)
    M = material_attenuations

    # sUnbinned: (180 x nEnergies) = detectorResponse .* incidentSpectrum'
    s_unbinned = detector_response * incident_spectrum.T

    # Define thresholds and binning (nBins x nEnergies)
    S = np.vstack([
        np.sum(s_unbinned[29:50, :], axis=0),
        np.sum(s_unbinned[50:61, :], axis=0),
        np.sum(s_unbinned[61:71, :], axis=0),
        np.sum(s_unbinned[71:82, :], axis=0),
        np.sum(s_unbinned[82:, :], axis=0),
    ])
    
    np.savetxt("s_python.txt", S, fmt="%.16f", delimiter="\t")

    if object_size >= 8:
        # Define phantom structure
        bone_density = 1.92
        unit = object_size // 8  # Phantom unit division
    
        # Create phantom matrices
        water = np.zeros((object_size, object_size, object_size))
        water[unit:7*unit, unit:7*unit, unit:7*unit] = np.ones((6*unit, 6*unit, 6*unit))
        water[unit:7*unit, 2*unit:3*unit, 2*unit:3*unit] = 0  # Cavity inside water
        
        #print('water8:', water)
    
        bone = np.zeros((object_size, object_size, object_size))
        bone[unit:7*unit, 2*unit:3*unit, 2*unit:3*unit] = 1  # Bone region
        
        #print('bone8:',bone)
    
        # Multiply volume fractions with densities
        water *= 1  # Water density is 1
        bone *= bone_density  # Apply bone density
    
        # Save ground truth data
        scipy.io.savemat('water_groundtruth.mat', {'water': water})
        scipy.io.savemat('bone_groundtruth.mat', {'bone': bone})
    
        # Flatten for further calculations
        x = np.column_stack((bone.ravel(), water.ravel()))
        
    elif object_size <= 4:
        bone_density = 1.92
        unit = object_size // 4  # Phantom unit division
    
        # Create phantom matrices
        water = np.zeros((object_size, object_size, object_size))
        water[unit:3*unit, unit:3*unit, unit:3*unit] = np.ones((2*unit, 2*unit, 2*unit))
        water[unit:3*unit, unit:2*unit, unit:2*unit] = 0  # Cavity inside water
            
        bone = np.zeros((object_size, object_size, object_size))
        bone[unit:3*unit, unit:2*unit, unit:2*unit] = 1  # Bone region
            
        water *= 1  # Water density is 1
        bone *= bone_density  # Apply bone density
    
        # Flatten for further calculations
        x = np.column_stack((bone.ravel(), water.ravel()))
    # print("x", x)
    # np.save('x.npy', x)

    # Define number of projections
    n_proj = 100
    projAngles = np.arange(0, n_proj)*(180/n_proj)  # Projection angles
    # Generate forward projection matrix
    A = conebeamtomo3(object_size, projAngles, pixel_pitch, n_pixels_y, n_pixels_z)

    # Apply forward model
    y = noisyforwardmodel(x, A, S, M)

    return y, S, M, A

