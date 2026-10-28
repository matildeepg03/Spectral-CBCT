import numpy as np
from projections import fp
def weidingerforwardmodel(x, A, M, S, nPixelsPerProj, astra):
    """
    Weidinger Forward Model

    Parameters:
    - x: numpy array, The input model, initialized as zeros ((objectSize^3) x nMats)
    - A: numpy array, subset of the forward model (nPixels x (objectSize^3))
    - M: numpy array, The attenuation matrix (nEnergies x nMats)
    - S: numpy array, Photon counts (nBins x nEnergies)
    - nPixelsPerProj: int, The number of detection pixels per projection

    Returns:
    - forw: numpy array, The photon counts for each energy bin and each pixel (nBins x nPixels)
    - forwmu: numpy array, The derivative for each energy bin and each pixel (nBins x nPixels x nMats)
    - forwmumu: numpy array, The double derivative for each energy bin and each pixel (nBins x nPixels x nMats x nMats)
    """
    
    nPixels = A.shape[0]
    nBins, nEnergies = S.shape
    nMats = M.shape[1]
    
    # Compute forward projections
    # projs = A @ x  # (nPixels x nMats)
    projs = fp(A, x)
    
    # Compute attenuation factors
    #print('max exp:', np.max(-M @ projs.T))
    #print('min exp:', np.min(-M @ projs.T))
    
    factor = -M @ projs.T
    factor[factor > 700] = 700 

    attenuationFactors = np.exp(factor)  # (nEnergies x nPixels)
    # print(f"weidingerforwardmodel.py____Projs {astra}:\n bone: max = {min(projs[:, 0])} {projs[:, 0]}\n water: max = {min(projs[:, 1])} {projs[:, 1]}")
    # print(f"x: min = [{min(x[:,0])}, {min(x[:,1])}], max = [{max(x[:,0])}, {max(x[:,1])}]")
    # print(f"Projs: min = [{min(projs[:,0])}, {min(projs[:,1])}], max = [{max(projs[:,0])}, {max(projs[:,1])}]")
    # print(f"factor: min = {min((factor)[0,:])}, max = {max((factor)[0,:])}")
    # print(f"attenuationFactors: min = {min(attenuationFactors[0,:])}, max = {max(attenuationFactors[0,:])}")


    # Compute forward model photon counts
    forw = S @ attenuationFactors  # (nBins x nPixels)

    # print(f"forw: min = {min(forw[0,:])}, max = {max(forw[0,:])}")

    # Compute first derivative (forwmu)
    attenuationFactorsMu = attenuationFactors[:,:,None] * M.reshape(nEnergies, 1, nMats)  # (nEnergies x nPixels x nMats)
    forwmu = -S @ attenuationFactorsMu.reshape(nEnergies, -1)  # (nBins x (nPixels * nMats))
    forwmu = forwmu.reshape(nBins, nPixels, nMats)  # (nBins x nPixels x nMats)
    
    # Compute second derivative (forwmumu)
    attenuationFactorsMu_expanded=attenuationFactorsMu[:, :, :, np.newaxis]
    attenuationFactorsMuMu = np.reshape(M, (nEnergies, 1, 1, nMats)) * attenuationFactorsMu_expanded  # (nEnergies x nPixels x nMats x nMats)
    forwmumu = S @ attenuationFactorsMuMu.reshape(nEnergies, -1)  # (nBins x (nPixels * nMats * nMats))
    forwmumu = forwmumu.reshape(nBins, nPixels, nMats, nMats)  # (nBins x nPixels x nMats x nMats)

    return forw, forwmu, forwmumu