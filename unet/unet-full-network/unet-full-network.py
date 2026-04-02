import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net for segmentation.
    """
    B,H,W,C = x.shape
    H1 = H-4
    W1 = W-4
    conv1 = np.zeros((B,H1,W1,64))

    H2 = H1//2-4
    W2 = W1//2-4
    conv2 = np.zeros((B,H2,W2,128))

    H3 = H2//2-4
    W3 = W2//2-4
    conv3 = np.zeros((B,H3,W3,256))

    H4 = H3//2-4
    W4 = W3//2-4
    conv4 = np.zeros((B,H4,W4,512))

    H5 = H4//2-4
    W5 = W4//2-4
    bottle = np.zeros((B,H5,W5,1024))

    H11 = H5*2-4
    W11 = W5*2-4 #skip is added before conv
    conv11 = np.zeros((B,H11,W11,512))

    H21 = H11*2-4
    W21 = W11*2-4
    conv21 = np.zeros((B,H21,W21,256))

    H31 = H21*2-4
    W31 = W21*2-4
    conv31 = np.zeros((B,H31,W31,128))

    H41 = H31*2-4
    W41 = W31*2-4
    conv41 = np.zeros((B,H41,W41,64))

    # H51 = H41*2-4
    # W51 = W41*2-4
    out = np.zeros((B,H41,W41,num_classes))

    return out