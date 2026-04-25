import numpy as np

class CrossEntropyError:
    def forward(self, y, t):
        # y: 예측, t: 정답
        return -np.sum(t * np.log(y + 1e-7)) / y.shape[0]
        
    def backward(self, y, t):
        batch_size = t.shape[0]
        return (-t / (y + 1e-7)) / batch_size
        
        
class SoftmaxWithLoss:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.y = np.exp(x) / np.sum(np.exp(x))
        return self.y

    def backward(self, t):
        return self.y - t

    def update(self):
        pass

