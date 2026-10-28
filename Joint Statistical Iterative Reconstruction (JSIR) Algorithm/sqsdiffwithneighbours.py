import numpy as np

def sqsdiffwithneighbours(input_array):
    """
    Computes the difference with surrounding pixels.
    
    Assumes the input image is 4D (3D, channels) and returns 
    a 5D image (3D, channels, neighboring pixel). Each of the (:,:,:,?) 
    slices contains the difference between the center pixel and one of its neighbors.

    Parameters:
    - input_array: numpy array of shape (objectSize, objectSize, objectSize, nMats)
    
    Returns:
    - diffs: numpy array of shape (objectSize, objectSize, objectSize, nMats, 27)
    """

    N = 3
    x, y, z, channels = input_array.shape
    diffs = np.zeros((x, y, z, channels, N**3), dtype=input_array.dtype)

    pad_width = ((N//2, N//2), (N//2, N//2), (N//2, N//2), (0, 0))
    padded_input = np.pad(input_array, pad_width, mode='edge')

    count = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                diffs[:, :, :, :, count] = input_array - padded_input[i:i+x, j:j+y, k:k+z, :]
                count += 1

    return diffs
