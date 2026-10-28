import matplotlib.pyplot as plt


def profile(bone_slice, water_slice, objectSize, name, nIter, nSubset, save):


    mid = objectSize //2 - 2

    bone_line = bone_slice[mid,:]
    water_line = water_slice[mid,:]
    plt.figure(figsize = (10,6))
    plt.plot(bone_line, label = 'Bone', color = 'blue')
    plt.plot(water_line, label = 'Water', color = 'orange')

    plt.axhline(0, color='black', lw=1)  # Baseline
    plt.title(f"1D Profile {name} - Slice: {mid}")
    plt.xlabel("Pixel Index (Distance across object)")
    plt.ylabel("Reconstructed Intensity (Brightness)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    if save:
        plt.savefig(f'code_simple_water_bone/reconstruction_results/profile_{name}_2_slice{mid}.png')
    plt.show()