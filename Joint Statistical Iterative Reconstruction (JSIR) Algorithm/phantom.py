import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
phantom.py contains the functions for generating phantoms:
    - phantom_original(): this is the function used in the code previously
    
    - phantom_square(): this is an adjusted version of phantom_original() making it easier to adjust the sizes of the water and bone 'boxes' in the phantom. Several some basic phantoms have been implemented for easier testing purposes for objectSize 10 and 32:
    
        Phantom 0: water square, 1 bone square inside
        Phantom 1: water square, 2 bone square inside
        Phantom 2: small water square, small bone square, no overlap
        phantom 3: zero phantom
        phantom 4: like phantom 1 with smaller water square and bone squares shifted more to top left
        
    -phantom_print():  this function is used to print the phantom in one image together with the reconstructions by playiterates(). It can be used to check the phantom geometry, although it will show empty boxes (which will be filled with reconstructions)
"""

def phantom_original(object_size):
    x = []
    if object_size >= 8:
        # Define phantom structure
        bone_density = 1.92
        unit = object_size // 8  # Phantom unit division

        # Create phantom matrices
        water = np.zeros((object_size, object_size, object_size))
        water[unit:7 * unit, unit:7 * unit, unit:7 * unit] = np.ones((6 * unit, 6 * unit, 6 * unit))
        water[unit:7 * unit, 2 * unit:3 * unit, 2 * unit:3 * unit] = 0  # Cavity inside water

        # print('water8:', water)

        bone = np.zeros((object_size, object_size, object_size))
        bone[unit:7 * unit, 2 * unit:3 * unit, 2 * unit:3 * unit] = 1  # Bone region

        # print('bone8:',bone)

        # Multiply volume fractions with densities
        water *= 1  # Water density is 1
        bone *= bone_density  # Apply bone density

        # Save ground truth data
        scipy.io.savemat('water_groundtruth.mat', {'water': water})
        scipy.io.savemat('bone_groundtruth.mat', {'bone': bone})

        slice_index = (object_size // 2) - 1
        water_slice = water[slice_index, :, :].T
        bone_slice = bone[slice_index, :, :].T

        # plt.imshow(water_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
        # plt.title('Water Phantom')
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(bone_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
        # plt.title('Bone Phantom')
        # plt.colorbar()
        # plt.show()
        # Flatten for further calculations
        x = np.column_stack((bone.ravel(), water.ravel()))

    elif object_size <= 4:
        bone_density = 1.92
        unit = object_size // 4  # Phantom unit division

        # Create phantom matrices
        water = np.zeros((object_size, object_size, object_size))
        water[unit:3 * unit, unit:3 * unit, unit:3 * unit] = np.ones((2 * unit, 2 * unit, 2 * unit))
        water[unit:3 * unit, unit:2 * unit, unit:2 * unit] = 0  # Cavity inside water

        bone = np.zeros((object_size, object_size, object_size))
        bone[unit:3 * unit, unit:2 * unit, unit:2 * unit] = 1  # Bone region

        water *= 1  # Water density is 1
        bone *= bone_density  # Apply bone density
        slice_index = (object_size // 2) - 1
        water_slice = water[slice_index, :, :].T
        bone_slice = bone[slice_index, :, :].T

        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(water_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
        # plt.title('Water Phantom')
        # plt.colorbar()

        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(bone_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
        # plt.title('Bone Phantom')
        # plt.colorbar()
        # plt.show()
        # Flatten for further calculations
        x = np.column_stack((bone.ravel(), water.ravel()))

    return x


def phantom_square(object_size, idx_phantom):
    """
    This function generates a square phantom with bone in two locations.

    :param object_size: size of the object
    :return: bone and water phantom
    """
    bone_density = 1.92
    if object_size ==10:
        unit = object_size // 10  # Phantom unit division
    elif object_size ==32:
        unit = object_size // 8
    elif object_size ==3:
        unit = object_size // 3
    #phantom 1 x 10:
    bone_start_factor_x2, bone_start_factor_y2, bone_start_factor_z2 = 0, 0, 0
    bone_stop_factor_x2, bone_stop_factor_y2, bone_stop_factor_z2 = 0, 0, 0
    if idx_phantom == 0:
        if object_size ==10:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 3, 1, 1
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 9, 9

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 3, 3, 3
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 5, 5

        elif object_size == 32:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 1, 1, 1
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 7 , 7, 7

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 1, 2, 2
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 7, 3, 3

        elif object_size ==3:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 0, 1, 0
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 2, 3, 2

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 0, 1, 1
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 2, 2, 2

    elif idx_phantom == 1:
        if object_size == 10:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 3, 1, 1
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 9, 9

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 3, 3, 3
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 5, 5

            bone_start_factor_x2, bone_start_factor_y2, bone_start_factor_z2 = 3, 5, 5
            bone_stop_factor_x2, bone_stop_factor_y2, bone_stop_factor_z2 = 6, 7, 7

        elif object_size ==32:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 1, 1, 1
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 7, 7, 7

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 2, 3, 3
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 4, 4

            bone_start_factor_x2, bone_start_factor_y2, bone_start_factor_z2 = 2, 4, 4
            bone_stop_factor_x2, bone_stop_factor_y2, bone_stop_factor_z2 = 6, 5, 5

    elif idx_phantom == 2:
        if object_size == 10:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 3, 5, 5
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 6, 6

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 3, 4, 4
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 5, 5

        elif object_size ==32:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 3, 4, 4
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 5, 5

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 3, 3, 3
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 4, 4

    #
    # # phantom 2 x 10:
    elif idx_phantom == 3:
        if object_size == 10:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 6, 7, 7
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 7, 7

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 6, 5, 5
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 5, 5
        elif object_size ==32:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 6, 7, 7
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 7, 7

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 6, 4, 4
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 4, 4

    elif idx_phantom ==4:
        if object_size ==10:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 3, 1, 1
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 6, 8, 8

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 3, 2, 2
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 6, 4, 4

            bone_start_factor_x2, bone_start_factor_y2, bone_start_factor_z2 = 3, 5, 5
            bone_stop_factor_x2, bone_stop_factor_y2, bone_stop_factor_z2 = 6, 7, 7
        elif  object_size ==32:
            water_start_factor_x, water_start_factor_y, water_start_factor_z = 1, 1, 1
            water_stop_factor_x, water_stop_factor_y, water_stop_factor_z = 7, 6, 6

            bone_start_factor_x, bone_start_factor_y, bone_start_factor_z = 1, 2, 2
            bone_stop_factor_x, bone_stop_factor_y, bone_stop_factor_z = 7, 3, 3

            bone_start_factor_x2, bone_start_factor_y2, bone_start_factor_z2 = 1, 4, 4
            bone_stop_factor_x2, bone_stop_factor_y2, bone_stop_factor_z2 = 7, 5, 5


    start_water_x, start_water_y,start_water_z = water_start_factor_x * unit, water_start_factor_y * unit, water_start_factor_z * unit
    stop_water_x, stop_water_y, stop_water_z = water_stop_factor_x * unit, water_stop_factor_y * unit, water_stop_factor_z * unit
    start_bone_x, start_bone_y,start_bone_z = bone_start_factor_x * unit, bone_start_factor_y * unit, bone_start_factor_z * unit
    stop_bone_x, stop_bone_y, stop_bone_z =  bone_stop_factor_x * unit, bone_stop_factor_y * unit, bone_stop_factor_z * unit



    # Create phantom matrices
    water = np.zeros((object_size, object_size, object_size))
    water[start_water_x:stop_water_x, start_water_y:stop_water_y, start_water_z:stop_water_z] = np.ones(((water_stop_factor_x - water_start_factor_x) * unit, (water_stop_factor_y - water_start_factor_y) * unit, (water_stop_factor_z - water_start_factor_z) * unit)) #change
    water[start_bone_x:stop_bone_x, start_bone_y:stop_bone_y, start_bone_z: stop_bone_z] = 0  # Cavity inside water


    bone = np.zeros((object_size, object_size, object_size))
    bone[start_bone_x:stop_bone_x, start_bone_y:stop_bone_y, start_bone_z: stop_bone_z] = 1  # Bone region
    # bone[start_water_x:stop_water_x, start_water_y:stop_water_y, start_water_z: stop_water_z] = 0  # Cavity inside bone, only use if bone region larger than water region
    if bone_start_factor_x2 != 0:
        start_bone_x2, start_bone_y2, start_bone_z2 = bone_start_factor_x2 * unit, bone_start_factor_y2 * unit, bone_start_factor_z2 * unit
        stop_bone_x2, stop_bone_y2, stop_bone_z2 = bone_stop_factor_x2 * unit, bone_stop_factor_y2 * unit, bone_stop_factor_z2 * unit
        water[start_bone_x2:stop_bone_x2, start_bone_y2:stop_bone_y2, start_bone_z2: stop_bone_z2] = 0
        bone[start_bone_x2:stop_bone_x2, start_bone_y2:stop_bone_y2, start_bone_z2: stop_bone_z2] = 1
    # print('bone8:',bone)

    # # Multiply volume fractions with densities
    water *= 1  # Water density is 1
    bone *= bone_density  # Apply bone density


    return np.column_stack((bone.ravel(), water.ravel()))
def phantom_print(x, object_size, title):
    bone_flat = x[:, 0]
    water_flat = x[:, 1]

    # 2. Reshape back to 3D cubes
    bone_3d = bone_flat.reshape((object_size, object_size, object_size))
    water_3d = water_flat.reshape((object_size, object_size, object_size))
    slice_index = (object_size // 2) - 1

    water_slice = water_3d[slice_index, :, :].T
    bone_slice = bone_3d[slice_index, :, :].T
    fig, axes = plt.subplots(3, 2, figsize=(8,18))
    # plt.figure(figsize=(8, 10))
    fig.suptitle(title)


    Im2 = axes[0, 1].imshow(bone_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
    axes[0,1].set_title('Bone Phantom')
    fig.colorbar(Im2, ax=axes[0, 1], fraction=0.046, pad=0.04)


    Im1 = axes[0, 0].imshow(water_slice, cmap='gray', vmin=0, vmax=1, interpolation='none')
    axes[0, 0].set_title('Water Phantom')
    fig.colorbar(Im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    return fig, axes

"""
    Phantom 0: water square, 1 bone square inside
    Phantom 1: water square, 2 bone square inside
    Phantom 2: small water square, small bone square, no overlap
    phantom 3: zero phantom
    phantom 4: like phantom 1 with smaller water square and bone squares shifted more to top left
   """

if __name__ == '__main__':
    objectSize = 32
    idx_phantom = 0
    phantom_print(phantom_square(objectSize, idx_phantom),objectSize, "test")
    plt.show()
