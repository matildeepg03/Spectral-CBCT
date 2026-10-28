import numpy as np
"""
This file contains the functions for the forward and backprojection operations, which was necessary to use astra with multiple materials.
"""
def fp(A, x):
    fp = []
    for i in range(x.shape[1]):
        phantom = x[:,i]
        fp_temp = A @ phantom
        fp.append(fp_temp)
        forw_projection = np.column_stack(fp)
    return  forw_projection.astype(np.float64)
def bp(A, fp):
    bp = []
    for i in range(fp.shape[1]):
        bp_temp = A.T @ fp[:,i]
        bp.append(bp_temp)
        back_projection = np.column_stack(bp)
    return back_projection.astype(np.float64)


