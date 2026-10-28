import numpy as np
import sqsdiffwithneighbours
import huber
from projections import fp
import time
def sqscomputecost(x, measurements, A, M, S, lambda_, delta_huber, nMats, objectSize, astra):
    """
    Computes the cost function of SQS methods.

    Parameters:
    - x: numpy array, the object as measured ((objectSize^3) x nMats)
    - measurements: numpy array, noisy "actual" measurements of an object (nBins x nPixels)
    - A: numpy array, forward projection matrix (nPixels x (objectSize^3))
    - M: numpy array, material attenuation for each energy (nEnergies x nMats)
    - S: numpy array, bin photon count (nBins x nEnergies)
    - lambda_: numpy array, array of lambda parameters (1 x nMats)
    - delta_huber: numpy array, array of delta huber parameters (1 x nMats)
    - nMats: int, number of materials
    - objectSize: int, length of one side of a cubed object

    Returns:
    - totalCost: float, sum of data attachment cost and regularization cost
    - dataAttCost: float, data attachment cost
    - regCost: float, cost of the regularization part
    """

    # Forward projection
    # projs = A @ x  # (nPixels x nMats)
    projs = fp(A, x)

    # Compute attenuation factors
    attenuationFactors = np.exp(-M @ projs.T)  # (nEnergies x nPixels)

    # Compute expected photon counts
    expectedCounts = S @ attenuationFactors  # (nBins x nPixels)
    # Data attachment cost
    # dataAttCost = -np.sum(measurements * np.log(expectedCounts) - expectedCounts)
    dataAttCost = -np.sum(measurements * np.log(expectedCounts) - expectedCounts - measurements * np.log(measurements) + measurements)

    ## print statements for debugging:
    # print(f"sqscomputecost.py____Projs {astra}:\n bone: max = {max(projs[:, 0])} {projs[:, 0]}\n water: max = {max(projs[:, 1])} {projs[:, 1]}")
    # print(f"projs.shape: {projs.shape}")
    # print(f"x: min = [{min(x[:,0])}, {min(x[:,1])}], max = [{max(x[:,0])}, {max(x[:,1])}]")
    # print(f"Projs sqscomputecosts: min = [{min(projs[:,0])}, {min(projs[:,1])}], max = [{max(projs[:,0])}, {max(projs[:,1])}]")
    # print(f"M.shape():{M.shape}")
    # print(f"M: min = {min(M[:,0])}, max = {max(M[:,1])}")
    # print(f"-M @ projs.T: min = {min((-M @ projs.T)[0,:])}, max = {max((-M @ projs.T)[0,:])}")
    # print(f"attenuationFactors: min = {min(attenuationFactors[0,:])}, max = {max(attenuationFactors[0,:])}")
    # print(f"expectedCounts: min = {min(expectedCounts[0,:])}, max = {max(expectedCounts[0,:])}")
    # print(f"measurements: min = {min(measurements[0,:])}, max = {max(measurements[0,:])}")
    # print(f"expectedCounts-measurements: {min(expectedCounts[0,:]-measurements[0,:]) + max(expectedCounts[0,:]-measurements[0,:])}")


    # Reshape object for neighbor computation
    tofilter = x.reshape((objectSize, objectSize, objectSize, nMats))

    # Compute differences with neighbors
    diffs = sqsdiffwithneighbours.sqsdiffwithneighbours(tofilter)  # (objectSize, objectSize, objectSize, nMats, 27)

    # Compute Huber loss
    regulAllMats = np.sum(huber.huber(diffs, delta_huber, 0), axis=4)  # (objectSize, objectSize, objectSize, nMats)

    # Regularization cost
    regCost = 0
    for material in range(nMats):
        regCost += lambda_[material] * np.sum(regulAllMats[:, :, :, material])
    print(f"dataAttCost: {dataAttCost}")
    print("regCost",regCost)


    # Total cost
    totalCost = dataAttCost + regCost
    # print("totalCost",totalCost)


    return totalCost, dataAttCost, regCost