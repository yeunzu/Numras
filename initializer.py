import numpy as np

def Xavior(input_size, output_size):
    w = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
    return w
    
def He(input_size, output_size):
    w = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    return w