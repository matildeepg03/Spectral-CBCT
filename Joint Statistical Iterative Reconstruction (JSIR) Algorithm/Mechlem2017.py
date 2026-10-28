import numpy as np
from sqscomputecost import sqscomputecost
from weidingerforwardmodel import weidingerforwardmodel
from noisyforwardmodel import noisyforwardmodel
from sqsregul import sqsregul
from getnewtonupdate import getnewtonupdate
from projections import bp
import scipy.io
from scipy.sparse.linalg import aslinearoperator

np.set_printoptions(legacy='1.21')

from astra_file import get_astra_projection_matrix, generate_subsets_vecs, generate_subsets
import time

def mechlem2017(objectSize, y, S, M, A, nPixelsPerProj, nIter, nSubset, lambda_, delta_huber, angles, pixelPitch, nPixelsY, nPixelsZ, dSource, d_detector, astra, scale , nesterov = True):
    """
    Implements the Mechlem 2017 iterative image reconstruction method.

    Parameters:
        objectSize (int): Size of one side of a cubic object.
        y (ndarray): Photon counts for measurements (nBins x nPixels).
        S (ndarray): Bin photon count (nBins x nEnergies).
        M (ndarray): Material attenuation matrix (nEnergies x nMats).
        A (ndarray): Forward projection matrix (nPixels x (objectSize^3)).
        nPixelsPerProj (int): Number of detection pixels per projection.
        nIter (int): Number of iterations.
        nSubset (int): Number of subsets.
        lambda_ (ndarray): Array of lambda parameters.
        delta_huber (ndarray): Array of delta Huber parameters.

    Returns:
        iterates (ndarray): Iteration results ((objectSize^3) x nMats x nIter).
        costs (ndarray): Cost associated with the updated iterates.
    """

    # Number of materials
    nMats = M.shape[1]

    # Initialize variables
    alpha_k = np.zeros((objectSize ** 3, nMats))

    v_k = np.zeros_like(alpha_k)
    z_k = np.zeros_like(alpha_k)


    iterates = np.zeros((objectSize ** 3, nMats, nIter))
    costs = np.zeros(nIter)
    total_costs = []
    dataAtt_costs = []
    reg_costs = []
    times = []
    # Compute the number of projections
    nProj = A.shape[0] // nPixelsPerProj
    # Shuffle projection order for Ordered Subsets (OS)
    np.random.seed(12345) #mg: same seed for comparisons
    shuffled = np.random.permutation(nProj)
  
    map_ = np.kron(shuffled[:, None] * nPixelsPerProj, np.ones((nPixelsPerProj, 1))).astype(int)
    map_ += np.kron(np.ones((nProj, 1)), np.arange(nPixelsPerProj).reshape(-1, 1)).astype(int)

    if astra != 1:
        shuffledA = A[map_.flatten(), :]
        shuffledMeasurements = y[:, map_.flatten()]

    else:
        y_temp = y.reshape((S.shape[0], nPixelsZ, nProj, nPixelsY))


    # Build subset matrices
    subAs = [None] * nSubset
    subMeasurements = [None] * nSubset
    subAirScales = [None] * nSubset
    sum_aijs = [None] * nSubset
    # Compute projections per subset
    nProjectionsPerSubset = int(np.ceil(nProj / nSubset))

    for subset in range(nSubset):
        firstRow = subset * nProjectionsPerSubset * nPixelsPerProj
        if astra == 1:
            # create indices
            startProj = subset * nProjectionsPerSubset
            endProj = min((subset + 1) * nProjectionsPerSubset, nProj)
            sub_indices = shuffled[startProj:endProj]

            subAngles = angles[sub_indices]
            subAs[subset], sum_aijs[subset] = generate_subsets(subAngles, objectSize, pixelPitch, nPixelsY, nPixelsZ, dSource, d_detector, scale)


            shuffledMeasurements = y_temp[:,:,sub_indices,:]
            sub_meas = shuffledMeasurements.reshape((S.shape[0], -1))
            subMeasurements[subset] = np.ascontiguousarray(sub_meas)

        else:
            lastRow = min((subset + 1) * nProjectionsPerSubset * nPixelsPerProj, shuffledA.shape[0])
            subAs[subset] = shuffledA[firstRow:lastRow, :]
            sum_aijs[subset] = np.array(subAs[subset].sum(axis=1)).reshape(-1, 1)

            subMeasurements[subset] = shuffledMeasurements[:, firstRow:lastRow]

    # Compute Nesterov coefficients
    tNesterov = np.ones(nIter * nSubset + 1)
    sumNesterov = np.zeros(nIter * nSubset + 1)
    sumNesterov[0] = 1;  # added mg

    for k in range(1, nIter * nSubset + 1):
        tNesterov[k] = 0.5 * (1 + np.sqrt(1 + 4 * tNesterov[k - 1] ** 2))
        sumNesterov[k] = sumNesterov[k - 1] + tNesterov[k]

    ratiosNesterov = tNesterov / sumNesterov

    # Start iterations
    start_time = time.time()

    for iter in range(nIter):

        times.append(time.time() - start_time)

        for subset in range(0,nSubset):
            # Forward model computation
            forw, forwmu, forwmumu = weidingerforwardmodel(z_k, subAs[subset], M, S, nPixelsPerProj, astra)
            ratios = subMeasurements[subset] / (forw + np.finfo(float).eps)

            
            # Compute gradient
            toBackProject = (1 - ratios)[:,:,np.newaxis] * forwmu
            toBackProject = np.sum(toBackProject, axis=0)  # Sum over bins
            gradients = bp(subAs[subset], toBackProject)
            # Compute Hessian
            forwmumu_sum = np.sum(forwmumu,axis=0)
            forwmumu_sum_ext = np.array(forwmumu_sum[:,:,:,np.newaxis])
            sum_aijs[subset] = np.array(sum_aijs[subset])
            toBackProject = forwmumu_sum_ext * sum_aijs[subset][:, None, None]
            toBackProject = toBackProject.reshape((toBackProject.shape[0], nMats * nMats))

            hessians = bp(subAs[subset], toBackProject)

            regulGrad, regulHess = sqsregul(z_k, objectSize, lambda_, delta_huber, nMats, nSubset)
            
            # Apply regularization
            gradients += regulGrad
            g_k = getnewtonupdate(hessians, regulHess, gradients, nMats)

            i = (iter) * nSubset + subset 

            if nesterov:
                # Apply updates
                alpha_k = z_k - g_k
                v_k -= tNesterov[i] * g_k
                z_k = alpha_k + ratiosNesterov[i + 1] * (v_k - alpha_k)

            else:
                alpha_k = alpha_k - g_k
                z_k = alpha_k
        # Store iteration results
        iterates[:, :, iter] = z_k
    
    return iterates,  total_costs, dataAtt_costs, reg_costs, times

