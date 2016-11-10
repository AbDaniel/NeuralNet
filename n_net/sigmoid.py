import numpy as np
from scipy.special import expit

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
