import numpy as np
import huber
import sqsdiffwithneighbours

def sqsregul(input_array, objectSize, lambda_, delta_huber, nMats, nSubsets):
    """
    Computes the gradient and Hessian of the Huber regularization function.

    Parameters:
    - input_array: numpy array, the input object
    - objectSize: int, the length of one side of a cubed object
    - lambda_: numpy array, array of lambda parameters (1 x nMats)
    - delta_huber: numpy array, array of delta Huber parameters (1 x nMats)
    - nMats: int, number of materials
    - nSubsets: int, number of subsets

    Returns:
    - gradCost: numpy array, gradient of the Huber cost function ((objectSize^3) x nMats)
    - hessCost: numpy array, Hessian of the Huber cost function ((objectSize^3) x nMats)
    """

    # Reshape input array
    tofilter = input_array.reshape((objectSize, objectSize, objectSize, nMats))

    # Compute differences with neighbors
    diffs = sqsdiffwithneighbours.sqsdiffwithneighbours(tofilter)  # (objectSize, objectSize, objectSize, nMats, 27)

    # Compute Huber loss values
    valueCost = np.sum(huber.huber(diffs, delta_huber, 0), axis=4)  # (objectSize, objectSize, objectSize, nMats)
    gradCost = np.sum(huber.huber(diffs, delta_huber, 1), axis=4)  # (objectSize, objectSize, objectSize, nMats)
    hessCost = np.sum(huber.huber(diffs, delta_huber, 2), axis=4)  # (objectSize, objectSize, objectSize, nMats)

    # Multiply by regularization weights
    lambda_reshaped = lambda_.reshape((1, 1, 1, nMats))
    valueCost *= lambda_reshaped
    gradCost = 2 * gradCost * lambda_reshaped / nSubsets
    hessCost = 4 * hessCost * lambda_reshaped / nSubsets

    # Reshape outputs
    valueCost = np.sum(valueCost)  # Scalar value
    gradCost = gradCost.reshape((objectSize**3, nMats))  # (objectSize^3) x nMats
    hessCost = hessCost.reshape((objectSize**3, nMats))  # (objectSize^3) x nMats

    return gradCost, hessCost