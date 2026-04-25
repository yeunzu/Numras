import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads, *args):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
            
            
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads, *args):
        # TODO: 나중에 구현
        pass
    
    