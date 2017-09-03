from numpy.random import randn
import numpy as np
import pandas as pd

def fibonacci_data(N):
    """
    Generate Fibonacci data 
    
    Input
    ------
    N: Number of data points expceted in fibonacci data set
    
    Output
    -------
    fib_data: dataframe of length N*2 with x & y coordinate of Fibonacci data along with 1 or -1 lable 
                associated to each data point
    
    """
    
    np.random.seed(0)
    theta = randn(1,N)*np.log10(3.5*np.pi)
    theta = 10**theta + np.pi
    theta = 4*np.pi - theta
    a = 0.1
    b = 0.2
    r = a*np.exp(b*theta)

    x1 = r*np.cos(theta)
    y1 = r*np.sin(theta)
    d = 0.14*np.sqrt(x1**2 + y1**2)
    x1 = x1 + randn(1,N)*d
    y1 = y1 + randn(1,N)*d

    x2 = r*np.cos(theta + np.pi)
    y2 = r*np.sin(theta + np.pi)
    d = 0.14*np.sqrt(x1**2 + y1**2)
    x2 = x2 + randn(1,N)*d
    y2 = y2 + randn(1,N)*d

    x = list(x1[0]) + list(x2[0])
    y = list(y1[0]) + list(y2[0])

    fib_data = pd.DataFrame()
    fib_data['x_cord'] = x
    fib_data['y_cord'] = y
    fib_data['label'] = [1]*N + [-1]*N
    
    return fib_data