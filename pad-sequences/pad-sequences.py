import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)
    if max_len:
        L = max_len
    elif N==0:
        L = 0
    else:
        L = max(len(seq) for seq in seqs)
    ans = np.full((N,L),pad_value)
    for i in range(N):
        for j in range(min(L,len(seqs[i]))):
            ans[i][j] = seqs[i][j]
    return ans