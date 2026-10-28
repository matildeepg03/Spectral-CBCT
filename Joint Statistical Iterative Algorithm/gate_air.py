import numpy as np
# open_low/open_high shape: (nz, nProj, ny)
def gate_air(n_proj):
    """
    Load and process air sinograms from GATE simulation files.

    Returns:
    - gate_air: numpy array of shape (2, nPixels) containing the mean air sinograms for both energy bins
    """
    open_low = np.load("low_7.npy")
    open_high = np.load("high_7.npy")

    mean_air_low = np.mean(open_low, axis=1)    # (nz, ny)
    mean_air_high = np.mean(open_high, axis=1)  # (nz, ny)

    # Use the same air estimate for every phantom projection
    mean_air_low = np.repeat(
        mean_air_low[:, None, :],
        n_proj,
        axis=1
    )

    mean_air_high = np.repeat(
        mean_air_high[:, None, :],
        n_proj,
        axis=1
    )

    gate_air = np.stack(
        [mean_air_low.reshape(-1), mean_air_high.reshape(-1)],
        axis=0
    )
# shape: (2, nPixels)

    return gate_air