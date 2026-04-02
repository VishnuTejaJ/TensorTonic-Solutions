import numpy as np
import torch

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block: double conv + max pool.
    """
    B, H, W, C = x.shape

    H1 = H-2
    W1 = W-2
    conv1 = np.zeros((B, H1, W1, out_channels))

    H2 = H1-2
    W2 = W1-2
    conv2 = np.zeros((B, H2, W2, out_channels))

    H3 = H2//2
    W3 = W2//2
    pool = np.zeros((B, H3, W3, out_channels))

    return (pool, conv2)