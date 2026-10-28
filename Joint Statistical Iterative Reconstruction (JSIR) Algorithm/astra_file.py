import astra
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import aslinearoperator
from noisyforwardmodel import noisyforwardmodel
from phantom import phantom_square
import time
import scipy.sparse
from scipy.sparse import csc_matrix, hstack
def get_astra_projection_matrix(objectSize, pixelPitch, nPixelsY, nPixelsZ, projAngles, dSource, d_detector, scale):
    """"
    This function generates an astra CUDA projector based geometry defined in start.py
    """
    # create the working area of the volume of the object
    #these lines are for adjusting to the right unit
    total_width_cm = objectSize * scale
    lim = total_width_cm / 2
    vol_geom = astra.create_vol_geom(
        objectSize, objectSize, objectSize,
        -lim, lim, -lim, lim, -lim, lim
    )

    # create the pixel-source setup
    proj_geom = astra.create_proj_geom('cone', pixelPitch*scale, pixelPitch*scale, nPixelsZ, nPixelsY, projAngles, dSource/10,
                                       d_detector/10)
                            
    

    # Cannot run this on macOS, requires NVIDIA GPU
    projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
   
    W = astra.OpTomo(projector_id)

    vector_full = astra.geom_2vec(proj_geom)
    vectors = vector_full['Vectors']
    # Shuffle the vectors using the EXACT same index array you used for y!
    # shuffled_vectors = vectors_full[shuffled, :]


    return W, vol_geom, proj_geom, vectors




def generate_subsets(angles, objectSize, pixelPitch, nPixelsY, nPixelsZ, dSource, d_detector, scale):

    total_width_cm = objectSize * scale
    lim = total_width_cm / 2
    vol_geom = astra.create_vol_geom(
        objectSize, objectSize, objectSize,
        -lim, lim, -lim, lim, -lim, lim
    )

    # create the pixel-source setup, incl. rotations around specified centre
    proj_geom = astra.create_proj_geom('cone', pixelPitch * scale, pixelPitch * scale, nPixelsZ, nPixelsY,
                                       angles, dSource/10,
                                       d_detector/10)

    # Cannot run this on macOS, requires NVIDIA GPU
    projector_id_sub = astra.create_projector('cuda3d', proj_geom, vol_geom)
    W = astra.OpTomo(projector_id_sub)

    subAs = aslinearoperator(W)


    ones_vol = np.ones(objectSize ** 3)
    sum = subAs @ ones_vol
    sum_aijs = np.array(sum).reshape(-1, 1)
    return subAs, sum_aijs


def generate_subsets_vecs(sub_vectors, objectSize, nPixelsZ, nPixelsY, scale):
    """
    Generate subsets using the projection geometry vectors, slower than actual generate_subsets()

    """
    total_width_cm = objectSize * scale
    lim = total_width_cm / 2
    vol_geom = astra.create_vol_geom(
        objectSize, objectSize, objectSize,
        -lim, lim, -lim, lim, -lim, lim
    )


    proj_geom_sub = astra.create_proj_geom('cone_vec',  nPixelsZ, nPixelsY, sub_vectors)
    # Cannot run this on macOS, requires NVIDIA GPU
    projector_id_sub = astra.create_projector('cuda3d', proj_geom_sub, vol_geom)
    W = astra.OpTomo(projector_id_sub)
    subAs = aslinearoperator(W)

    ones_vol = np.ones(objectSize ** 3)
    sum = subAs @ ones_vol
    sum_aijs = np.array(sum).reshape(-1, 1)
    return subAs, sum_aijs

def matrix_generation(W, objectSize):
    W_matrix = []
    for i in range(objectSize ** 3):
        vec = np.zeros(objectSize ** 3)
        vec[i] = 1.0
        fp = W @ vec
        sparse_col = csc_matrix(fp.reshape(-1, 1), dtype=np.float64)
        W_matrix.append(sparse_col)
    W = hstack(W_matrix)
    return W



if __name__ == '__main__':
    objectSize = 32
    nPixelsY = 64
    nPixelsZ = 44
    pixelPitch = 1
    n_proj = 100
    projAngles = np.arange(0, n_proj) * (np.pi / n_proj)
    d_detector = 900
    dSource = 2300
    idx_phantom =0
    phantom_data = phantom_square(32, idx_phantom)
    W,_,_,_ = get_astra_projection_matrix(objectSize, pixelPitch, nPixelsY, nPixelsZ, projAngles, dSource, d_detector,phantom_data)
    print(phantom_data.shape)
    fp_astra_bone = W @ np.ones(phantom_data.shape[0])
    print(W.shape)

    n_voxels = objectSize**3
    n_pixels = nPixelsY * nPixelsZ * n_proj

    A = []
    indices = []
    index_pointer = [0]
    vector = np.zeros(n_voxels, dtype=np.float32)
    timelimit = 150
    starttime = time.time()

    for i in range(n_voxels):
    #     # print(i)
    #     if (time.time() - starttime) > timelimit:
    #         print("Time limit exceeded")
    #         break
        vector[i] = 1
        fp = W @ vector
        vector[i] = 0
        threshold = 1e-6
        test = np.abs(fp) > threshold
        column_values = fp[test]
        column_indices = np.where(test)[0]
        # print(fp)
        A.extend(column_values)
        indices.extend(column_indices)
        index_pointer.append(len(A))
    A_output = scipy.sparse.csc_matrix((A, indices, index_pointer), shape=(n_pixels, n_voxels))
    print(A_output)







