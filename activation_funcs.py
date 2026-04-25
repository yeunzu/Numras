import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
        
    def backward(self, dout):
        return dout * (self.y * (1 - self.y))
        
    def update(self):
        pass
    
    
class Tanh:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        self.y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        
    def backward(self, dout):
        return dout * (1 - self.y**2)
        
    def update(self):
        pass
    

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        return dout
        
    def update(self):
        pass
    
    
class Softmax:
    def __init__(self):
        self.y = None
        
    def forward(self, x):
        # 오버플로 방지 및 배치 처리 위해 axis=1 적용
        c = np.max(x, axis=1, keepdims=True)
        exp_a = np.exp(x - c)
        sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
        self.y = exp_a / sum_exp_a
        return self.y
        
    def backward(self, dout):
        # 역전파에도 각 데이터 단위로 게산되도록 axis=1 적용
        return self.y * (dout - np.sum(dout * self.y, axis=1, keepdims=True))
        
    def update(self):
        pass
    

