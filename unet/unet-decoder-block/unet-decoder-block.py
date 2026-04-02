import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net decoder block: up-conv + concat + double conv.
    """
    B, H, W, C = x.shape

    H1 = H*2
    W1 = W*2
    up_conv = np.zeros((B, H1, W1, C))

    C1 = C*2
    concated = np.zeros((B, H1, W1, C1))

    H2 = H1-2
    W2 = W1-2
    conv1 = np.zeros((B, H2, W2, out_channels))

    H3 = H2-2
    W3 = W2-2
    conv2 = np.zeros((B, H3, W3, out_channels))

    return conv2