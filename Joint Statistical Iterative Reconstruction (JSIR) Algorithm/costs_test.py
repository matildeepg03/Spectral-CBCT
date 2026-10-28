import numpy as np
from projections import fp
from simulatepreps_phantom import simulatepreps
import csv
def dataAttCost_test(A, x, y_meas, M, S, astra = 0):
    """"
    This function compares the cost when including the 'constant' term vs without to determine why the first value is very negative.
    """
    projs = fp(A, x)


    attenuationFactors = np.exp(-M @ projs.T)  # (nEnergies x nPixels)

    expectedCounts = S @ attenuationFactors  # (nBins x nPixels)
    dataAttCost = -np.sum(y_meas * np.log(expectedCounts) - expectedCounts )
    dataAttCost_cte = -np.sum(y_meas * np.log(expectedCounts) - expectedCounts -y_meas*np.log(y_meas)+y_meas)

    # np.savetxt("expectedCounts.csv", expectedCounts, delimiter=",", fmt='%f')
    #
    # # Save Zeros Cost
    # np.savetxt("measuredCounts.csv", y_meas, delimiter=",", fmt='%f')
    return dataAttCost, dataAttCost_cte

def save_matrix(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data) # 'writerows' handles the 2D list automatically

if __name__ == '__main__':
    objectSize = 10

    nIter = 15

    nSubset = 1

    # nPixelsY = 64
    # nPixelsZ = 44
    nPixelsY = 20
    nPixelsZ = 20

    # Pixel pitch
    pixelPitch = 1
    d_detector = 900
    dSource = 2300
    n_proj = 100
    projAngles = np.arange(0, n_proj) * (np.pi / n_proj)
    # The total number of pixels on the detector plate
    nPixelsPerProj = nPixelsZ * nPixelsY
    astra = 1  # 0: conebeamtomo3, 1: astra matrix, 2: astra projector
    scale = 0.1
    y, S, M, A, x = simulatepreps(objectSize, nPixelsY, nPixelsZ, pixelPitch, d_detector, dSource, projAngles, astra,
                                  scale,0)
    dataAttCost_zeros, dataAttCost_zeros_cte = dataAttCost_test(A, np.zeros((objectSize**3, 2)), y, M, S, astra)
    dataAttCost_correct,  dataAttCost_correct_cte= dataAttCost_test(A, x, y, M, S, astra)

    print('dataAttCost correct: ', dataAttCost_correct)
    print('dataAttCost zeros: ', dataAttCost_zeros)
    print('dataAttCost correct including constant measurement term: ', dataAttCost_correct_cte)
    print('dataAttCost zeros including constant measurement term : ', dataAttCost_zeros_cte)
    print('Abs difference dataAttCost correct vs zeros: ', np.abs(dataAttCost_correct-dataAttCost_zeros))




