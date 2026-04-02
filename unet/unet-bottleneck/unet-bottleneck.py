import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    """
    B,H,W,C = x.shape

    H1 = H-2
    W1 = W-2
    conv2 = np.zeros((B,H1,W1,out_channels))

    H2 = H1-2
    W2 = W1-2
    conv2 = np.zeros((B,H2,W2,out_channels))

    return conv2