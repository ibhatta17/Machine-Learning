import numpy as np
import pandas as pd

def xor_data(N):
    """
    Generate XOR data 
    
    Input
    ------
    N: Number of data points expceted in XOR data set
    
    Output
    -------
    xor: XOR data of length N
    
    """
    
    np.random.seed(0)
    X = np.random.randn(N, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1]> 0)
    y = np.where(y, -1, 1)
    xor = pd.DataFrame(X, columns = ['x_cord', 'y_cord'])
    xor['label'] = y
    
    return xor