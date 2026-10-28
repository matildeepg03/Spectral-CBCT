import numpy as np

def huber(input_array, delta_huber, order):
    """
    Computes the order-th derivative of the Huber function.

    Parameters:
    - input_array: numpy array of shape (objectSize, objectSize, objectSize, nMats, 27),
                   representing object differences.
    - delta_huber: numpy array, array of delta huber parameters (nMats,).
    - order: int, the order of the derivative (0, 1, or 2).

    Returns:
    - output: numpy array of the same shape as input_array, containing the ordered derivative
              of the Huber function.
    """

    # Ensure the number of materials matches the length of delta_huber
    _, _, _, channels, _ = input_array.shape
    if channels != len(delta_huber):
        print("Inconsistent delta and input")
        return None

    # Reshape delta_huber for broadcasting
    d = delta_huber.reshape((1, 1, 1, channels, 1))

    # Compute Huber function based on order
    if order == 0:
        output = 2 * d * np.abs(input_array) - d**2
        mask = np.abs(input_array) < d
        output[mask] = input_array[mask] ** 2

    elif order == 1:
        output = 2 * d * np.sign(input_array)
        mask = np.abs(input_array) < d
        output[mask] = 2 * input_array[mask]

    elif order == 2:
        output = np.zeros_like(input_array)
        mask = np.abs(input_array) < d
        output[mask] = 2

    else:
        print("Unhandled order")
        output = np.zeros_like(input_array)

    return output