import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features and concatenate with decoder features.
    """
    B1,H1,W1,C1 = encoder_features.shape
    B2,H2,W2,C2 = decoder_features.shape
    return np.concatenate((encoder_features[:,H1//2-H2//2:H1//2+H2//2,W1//2-W2//2:W1//2+W2//2,:],decoder_features),axis=-1)