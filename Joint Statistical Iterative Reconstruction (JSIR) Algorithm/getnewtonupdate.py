import numpy as np

def getnewtonupdate(hessians, regulHess, gradients, nMats):
    """
    Computes the update step of Newton's method, voxel-per-voxel,
    given the Hessians and the gradients.

    Parameters:
    - hessians: numpy array of shape ((objectSize^3) x (nMats * nMats)), Hessian matrix
    - regulHess: numpy array of shape ((objectSize^3) x nMats), Hessian of the regularization term
    - gradients: numpy array of shape ((objectSize^3) x nMats), combined gradient and regularized gradient
    - nMats: int, number of materials

    Returns:
    - updates: numpy array of shape ((objectSize^3) x nMats), Newton updates for each voxel.
    """

    updates = np.zeros_like(gradients)
    #print(updates.shape)

    # Compute f_n+1, voxel by voxel
    for vox in range(hessians.shape[0]):
        # Reshape hessians from (1, nMats*nMats) to (nMats, nMats)
        H = hessians[vox, :].reshape(nMats, nMats) + np.diag(regulHess[vox, :])

        # Compute Newton update using np.linalg.solve() for numerical stability
        updates[vox, :] = np.linalg.solve(H, gradients[vox, :])
        
    return updates
