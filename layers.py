import numpy as np
from optimizer import *
from initializer import *
from funcs_for_cnn import *
from activation_funcs import *

class Affine:
    def __init__(
        self,
        input_size=None,
        output_size=None,
        initializer=He,
        optimizer=SGD,
        learning_rate=0.01,
        **opt_params
        ):
        
        self.W = initializer(input_size, output_size)
        self.B = np.zeros(output_size)
        self.params = [self.W, self.B]
        
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        self.grads = [self.dW, self.dB]
        
        self.optimizer = optimizer(lr=learning_rate)
        self.opt_params = opt_params
        self.lr = learning_rate
        self.x = None
        
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.B
        return out
        
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis=0)
        self.grads[0] = self.dW
        self.grads[1] = self.dB
        return dx
        
    def update(self):
        self.optimizer.update(self.params, self.grads)
        
        
class Flatten:
    def __init__(self):
        self.original_x_shape = None
        
    def forward(self, x):
        self.original_x_shape = x.shape
        
        if x.ndim > 2:
            out = x.reshape(x.shape[0], -1)
        else:
            out = x.reshape(1, -1)
        
        return out
        
    def backward(self, dout):
        dx = dout.reshape(*self.original_x_shape)
        return dx
        
    def update(self):
        pass
    
    
class Conv:
    def __init__(self,
                 filter_num, filter_channels, filter_height, filter_width,
                 stride=1, pad=0, 
                 initializer=He, optimizer=SGD, learning_rate=0.01,
                 **opt_params
                 ):
        self.stride = stride
        self.pad = pad
        
        fan_in = filter_channels * filter_height * filter_width
        
        # W를 2차원으로 초기화 -> 4차원으로 형상 변환
        weight_2d = initializer(fan_in, filter_num)
        self.W = weight_2d.reshape(filter_channels, filter_height, filter_width, filter_num).transpose(3, 0, 1, 2)
        # 편향 개수 = 필터 개수
        self.B = np.zeros(filter_num)
        self.params = [self.W, self.B]
        
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        self.grads = [self.dW, self.dB]
        
        self.optimizer = optimizer(lr=learning_rate)

        # 역전파 시 중간 데이터 저장용
        self.x = None
        self.col = None
        self.col_W = None
        
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad) # 4차원 -> 2차원
        col_W = self.W.reshape(FN, -1).T # 필터도 2차원으로(FN, C, FH, FW) -> (FN, C*FH*FW) -> (C*FH*FW, FN)
        
        out = np.dot(col, col_W) + self.B
        
        # 2차원 -> 4차원 / 축 순서 (N, H, W, C) -> (N, C, H, W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        # 역전파를 위해 2차원으로 편 데이터 저장
        self.x = x
        self.col = col
        self.col_W = col_W
    
        return out
        
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
    
        # 4차원 -> 2차원
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.dB = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) # dW를 원래 4차원 필터로
        dcol = np.dot(dout, self.col_W.T) # 아래로 흘려보낼 데이터 기울기
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) # 2차원 -> 4차원
        
        self.grads[0] = self.dW
        self.grads[1] = self.dB
        
        return dx
        
    def update(self):
        self.optimizer.update(self.params, self.grads)
        
        
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) # 2차원으로
        col = col.reshape(-1, self.pool_h * self.pool_w) # 채널별로 최댓값을 구할 수 있는 형태로
        self.arg_max = np.argmax(col, axis=1) # 최댓값의 인덱스 저장(역전파에서 어디로 기울기를 보낼지 기억하기 위함)
        out = np.max(col, axis=1) # 채널별/영역별 최댓값
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # 원래의 4차원 형태로 변환
        self.x = x

        return out

    def backward(self, dout):
        # 역전파 => 최댓값에 있던 자리에만 기울기 흘리기
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))

        # arg_max -> 1차원 배열 형태에서 최댓값 있던 위치에만 dout 값 넣기
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))

        # 2차원으로 펼친 형태 -> 4차원으로 되돌리기
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

    def update(self):
        pass
    
    
class ResidualBlock:
    def __init__(self, filter_num, filter_height, filter_width, stride=1, pad=1, initializer=He, optimizer=SGD, lr=0.01):
        # 블록 내부에 사용할 레이어 조립
        # 입력 채널과 출력 채널이 같다고 가정할 때의 구성
        self.conv1 = Conv(filter_num, filter_num, filter_height, filter_width, stride, pad, initializer=initializer, optimizer=optimizer, learning_rate=lr)
        self.relu1 = Relu()
        self.conv2 = Conv(filter_num, filter_num, filter_height, filter_width, 1, pad, initializer=initializer, optimizer=optimizer, learning_rate=lr)
        self.internal_layers = [self.conv1, self.conv2] # 가중치 업데이트 위해 레이어 리스트로 묶기

    def forward(self, x):
        self.x = x # 지름길로 보낼 입력값 보관

        # 메인 경로 연산
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)

        # 지름길 더하기(F(x)+x)
        return out + self.x

    def backward(self, dout):
        # 메인 경로 역전파
        dx_main = self.conv2.backward(dout)
        dx_main = self.relu1.backward(dx_main)
        dx_main = self.conv1.backward(dx_main)

        # 최종 입력값 x로 흐르는 기울기 = (메인 경로 기울기 + 지름길 기울기)
        return dx_main + dout

    def update(self):
        for layer in self.internal_layers:
            layer.update()


class Affine_ResNet:
    def __init__(self, initializer=He, optimizer=SGD, learning_rate=0.01, size=None):
        self.affine1 = Affine(input_size=size, output_size=size, initializer=initializer, optimizer=optimizer, learning_rate=learning_rate)
        self.relu1 = Relu()
        self.affine2 = Affine(input_size=size, output_size=size, initializer=initializer, optimizer=optimizer, learning_rate=learning_rate)
        self.internal_layers = [self.affine1, self.affine2]

    def forward(self, x):
        self.x = x

        out = self.affine1.forward(x)
        out = self.relu1.forward(out)
        out = self.affine2.forward(out)

        return out + self.x

    def backward(self, dout):
        dx_main = self.affine2.backward(dout)
        dx_main = self.relu1.backward(dx_main)
        dx_main = self.affine1.backward(dx_main)

        return dx_main + dout

    def update(self):
        for layer in self.internal_layers:
            layer.update()


class BatchNormalization:
    def __init__(self, gamma=1.0, beta=0.0, momentum=0.9, running_mean=None, running_var=None, optimizer=SGD, learning_rate=0.01):
        self.gamma = gamma # 스케일(scale) 파라미터
        self.beta = beta   # 시프트(shift) 파라미터
        self.momentum = momentum
        self.input_shape = None # Conv 레이어 대응용 (4차원의 경우)

        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        # 시험 때 사용할 이동 평균 및 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # 역전파 시 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None
        
        # 파라미터 및 기울기
        self.params = [self.gamma, self.beta]
        self.grads = [self.dgamma, self.dbeta]

        self.train_flg = True
        
        self.optimizer = optimizer(lr=learning_rate)

    def forward(self, x, train_flg=None):
        if train_flg is None: train_flg=self.train_flg
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            D = x.shape[1]
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.grads[0] = dgamma
        self.grads[1] = dbeta
        
        return dx

    def update(self):
        # gamma와 beta 업데이트
        # 주의: self.params를 [self.gamma, self.beta]로 묶었으므로 리스트 내부 값이 업데이트 되어야 함
        self.optimizer.update(self.params, self.grads)


class ResidualBlock_With_Norm:
    def __init__(self, filter_num, filter_height, filter_width, stride=1, pad=1, initializer=He, optimizer=SGD, lr=0.01, train_flg=True):
        # 블록 내부에 사용할 레이어 조립
        # 입력 채널과 출력 채널이 같다고 가정할 때의 구성
        self.train_flg = train_flg
        self.conv1 = Conv(filter_num, filter_num, filter_height, filter_width, stride, pad, initializer=initializer, optimizer=optimizer, learning_rate=lr)
        self.bn1 = BatchNormalization(learning_rate=lr)
        self.relu1 = Relu()
        self.conv2 = Conv(filter_num, filter_num, filter_height, filter_width, 1, pad, initializer=initializer, optimizer=optimizer, learning_rate=lr)
        self.bn2 = BatchNormalization(learning_rate=lr)
        self.internal_layers = [self.conv1, self.bn1, self.conv2, self.bn2] # 가중치 업데이트 위해 레이어 리스트로 묶기

    def forward(self, x):
        self.bn1.train_flg = self.train_flg
        self.bn2.train_flg = self.train_flg
        self.x = x # 지름길로 보낼 입력값 보관

        # 메인 경로 연산
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        # 지름길 더하기(F(x)+x)
        return out + self.x

    def backward(self, dout):
        # 메인 경로 역전파
        dx_main = self.bn2.backward(dout)
        dx_main = self.conv2.backward(dx_main)
        dx_main = self.relu1.backward(dx_main)
        dx_main = self.bn1.backward(dx_main)
        dx_main = self.conv1.backward(dx_main)

        # 최종 입력값 x로 흐르는 기울기 = (메인 경로 기울기 + 지름길 기울기)
        return dx_main + dout

    def update(self):
        for layer in self.internal_layers:
            layer.update()


class Dropout:
    def __init__(self, dropout_ratio=0.5, train_flg=True):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = train_flg

    def forward(self, x, train_flg=None):
        if train_flg == None: train_flg = self.train_flg
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    
    def update(self, *args):
        pass