import numpy as np

def cosine_distnace(v1, v2):
    cos = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos

def euclidean_distance(v1, v2):
    euc = np.sqrt(np.sum(np.square(v1 - v2)))
    return euc
