import numpy as np
import scipy.io
import matplotlib.pyplot as plt
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
        water = np.random.rand(object_size, object_size, object_size)
        
        print('water8:', water)
    
        bone = np.random.rand(object_size, object_size, object_size)
        
        print('bone8:',bone)
    
        # Multiply volume fractions with densities
        water *= 1  # Water density is 1
        bone *= bone_density  # Apply bone density
        
        print('water check: ', water[0][0][0])
        print('bone check: ', bone[0][0][0])
        
        # Save ground truth data
        # scipy.io.savemat('water_groundtruth_randomized.mat', {'water': np.transpose(water, (2, 1, 0))})
        # scipy.io.savemat('bone_groundtruth_randomized.mat', {'bone': np.transpose(bone, (2, 1, 0))})
        water = scipy.io.loadmat('/Users/MaxineVanHoutven1/PycharmProjects/maxine-project/water_groundtruth_randomized.mat')['water']
        bone = scipy.io.loadmat('/Users/MaxineVanHoutven1/PycharmProjects/maxine-project/bone_groundtruth_randomized.mat')['bone']
        slice_index = (object_size // 2) - 1
        water_slice = water[:, :, slice_index]
        bone_slice = bone[:, :, slice_index]

        plt.imshow(water_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
        plt.title('Water Phantom')
        plt.colorbar()
        plt.show()

        plt.imshow(bone_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
        plt.title('Bone Phantom')
        plt.colorbar()
        plt.show()
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

    # Define number of projections
    n_proj = 100
    projAngles = np.arange(0, n_proj)*(180/n_proj)  # Projection angles
    # Generate forward projection matrix
    A = conebeamtomo3(object_size, projAngles, pixel_pitch, n_pixels_y, n_pixels_z)

    # Apply forward model
    y = noisyforwardmodel(x, A, S, M)
    print("y", y)

    return y, S, M, A
