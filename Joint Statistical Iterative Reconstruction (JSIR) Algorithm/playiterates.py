import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from plot_profile import profile

def playiterates(iterates, mat_inf, mat_sup, objectSize, name, nIter, nSubset, save, astra, file_name, fig, axes):
    """
    Show the iterates.

    Parameters:
    - iterates: numpy array of size (objectSize^3, nMats, nIterates), reconstruction results.
    - mat_inf: numpy array, lower bound values for materials.
    - mat_sup: numpy array, upper bound values for materials.
    - objectSize: int, size of one side of the cubic object.

    Outputs:
    - Saves images of reconstructions.
    - Saves the matrix data in a .mat file.
    """

    _, nMats, nIterates = iterates.shape

    # Reshape iterates
    matres = iterates.reshape((objectSize, objectSize, objectSize, nMats, nIterates), order = 'F')
    matres1 = matres[:, :, (objectSize // 2) - 1, 0, -1]  # First material
    matres2 = matres[:, :, (objectSize // 2) - 1, 1, -1]  # Second material
    np.save("bone_map.npy", matres1)
    np.save("water_map.npy", matres2)

    # Display images
  
    Im2 = axes[astra+1, 1].imshow(matres1, cmap='gray', vmin=0, vmax=1, interpolation='none')
    axes[astra+1,1].set_title(f'Maybe bone {name}')
    fig.colorbar(Im2, ax=axes[astra+1, 1], fraction=0.046, pad=0.04)


    Im1 = axes[astra+1, 0].imshow(matres2, cmap='gray', vmin=0, vmax=1, interpolation='none')
    axes[astra+1, 0].set_title(f'Maybe water {name}')
    fig.colorbar(Im1, ax=axes[astra+1, 0], fraction=0.046, pad=0.04)

    # save water
    fig_w = plt.figure()
    plt.imshow(matres2, cmap='gray', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label(r'Water coefficient, $c_{\mathrm{water}}$')
    plt.title("Water Material Map")
    fig_w.savefig(f'/scratch/matildegoncalv/reconstruction_water_{file_name}.png')
    plt.close(fig_w)

    # save bone
    fig_b = plt.figure()
    plt.imshow(matres1, cmap='gray', vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label(r'Bone coefficient, $c_{\mathrm{bone}}$')
    plt.title("Bone Material Map")
    fig_b.savefig(f'/scratch/matildegoncalv/reconstruction_bone_{file_name}.png')
    plt.close(fig_b)

    #plt.imshow(matres1, cmap='gray', vmin = 0, vmax = 1, interpolation='none')
    #plt.title(f'Maybe bone {name}')
    #plt.colorbar()
    #
    #plt.subplot(4, 2, n)
    #plt.imshow(matres2, cmap='gray', vmin = 0, vmax = 1, interpolation='none')
    #plt.title(f'Maybe water {name}')
    #plt.colorbar()


    """     if astra == 1:
        plt.tight_layout()
        if save:
            plt.savefig(f'/scratch/matildegoncalv/reconstruction_comparison/reconstruction_{file_name}.png')
        plt.show() """

    # can be used if only running one reconstruction or when reconstruction plotting turned off, otherwise it interferes
    # profile(matres1, matres2, objectSize, name, nIter, nSubset, save)
    return matres1, matres2


def watch_iteration_results(iterates, objectSize):
    """
    Plays a movie of the reconstruction iteration by iteration.
    """
    _, nMats, nIterates = iterates.shape

    # Reshape (Same logic as your code)
    matres = iterates.reshape((objectSize, objectSize, objectSize, nMats, nIterates), order='F')
    fig, axes = plt.subplots(7, 2, figsize=(8, 18))

    # Loop through every iteration
    for i in range(0,7):

        slice_bone = np.nan_to_num(matres[:, :, (objectSize // 2) - 1, 0, i])
        slice_water = np.nan_to_num(matres[:, :, (objectSize // 2) - 1, 1, i])

        # Clear and Plot Bone

        axes[i, 0].imshow(slice_bone, cmap='gray', vmin=0, vmax=1)  # Keep vmin/vmax fixed to see contrast changes!
        axes[i, 0].set_title(f"Bone - Iter {i}")

        # Clear and Plot Water

        axes[i, 1].imshow(slice_water, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Water - Iter {i}")

    plt.show()
