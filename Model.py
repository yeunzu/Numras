import numpy as np
from activation_funcs import *
from error_funcs import *
from initializer import *
from layers import *
from optimizer import *


class MyModel:
            def __init__(self):
                self.layer = []
                self.loss_func = None
                
                self.x_train = None
                self.y_train = None
                self.x_validation = None
                self.y_validation = None
                self.x_test = None
                self.y_test = None
        
                self.loss_log = []
                self.acc_log = []
        
            ## 데이터 처리
            def load_data(self, data_kind:str = None, x=None, y=None):
                if data_kind == 'train':
                    self.x_train, self.y_train = x, y
                elif data_kind == 'validation':
                    self.x_validation, self.y_validation = x, y
                elif data_kind == 'test':
                    self.x_test, self.y_test = x, y
                else:
                    # TODO: 에러 처리하기
                    pass
        
        
            def _data_batch_setting(self, _batch_size, _data):
                _data_size = _data.shape[0]
                if _data_size % _batch_size != 0:
                    raise ValueError("데이터 개수가 배치 사이즈로 나누어 떨어지지 않습니다.")
                
                batch_num = int(_data_size / _batch_size)
                # _data.shape[1:]는 (28, 28)이나 (10,) 같은 나머지 차원을 그대로 가져옵니다.
                # 즉, (60000, 28, 28) -> (600, 100, 28, 28) 로 한 번에 형태를 바꿉니다.
                return _data.reshape(batch_num, _batch_size, *_data.shape[1:])
        
        
            def data_batch_setting(self, batch_size, setting_data):
                if setting_data == 'x_train':
                    self.x_train = self._data_batch_setting(batch_size, self.x_train)
                elif setting_data == 'y_train':
                    self.y_train = self._data_batch_setting(batch_size, self.y_train)
                elif setting_data == 'x_validation':
                    self.x_validation = self._data_batch_setting(batch_size, self.x_validation)
                elif setting_data == 'y_validation':
                    self.y_validation = self._data_batch_setting(batch_size, self.y_validation)
                elif setting_data == 'x_test':
                    self.x_test = self._data_batch_setting(batch_size, self.x_test)
                elif setting_data == 'y_test':
                    self.y_test = self._data_batch_setting(batch_size, self.y_test)
                else:
                    # TODO: 에러 처리하기
                    pass
        
            ## 레이어 관리
            def add_layer(self, layer, sequence=None):
                if sequence == None:
                    self.layer.append(layer)
                else:
                    self.layer.insert(sequence, layer)
        
            def remove_layer(self, sequence):
                self.layer.pop(sequence)
        
            def set_loss_func(self, func):
                self.loss_func = func
        
            ## 아래부터 본격적인 순전파, 오차, 역전파, 갱신
            def predict(self, x):
                for layer in self.layer:
                    x = layer.forward(x)
                return x
        
            def get_loss(self, x, t):
                y = self.predict(x)
                loss = self.loss_func.forward(y, t)
                return y, loss
        
            def backward(self, y, t):
                self.layer.reverse() # 레이어 순서 뒤집어서 역전파가 진행되도록
                dout = self.loss_func.backward(y, t) # 손실함수에서 역전파의 시작으로 흘려보낼 값 구하기
                for layer in self.layer:
                    dout = layer.backward(dout)
                self.layer.reverse() # 원상복구
                return dout
        
            def update(self):
                for layer in self.layer:
                    layer.update()
        
            def learn(self, epoch):
                self.loss_log, self.acc_log = [], []
                for e in range(epoch): # 에포크만큼 반복
                    print(f"start epoch {e+1}")
                    # 정확도 & 손실
                    total_loss = 0
                    correct = 0
                    total_samples = 0
        
                    for i in range(len(self.x_train)): # 배치마다 도는 것도 가능하게끔
                        if len(self.x_train) == len(self.y_train):
                            pass
                        else:
                            # TODO: 에러 처리하기
                            print("x랑 y 개수 안맞음")
                            break
                        x, t = self.x_train[i], self.y_train[i] # 데이터 꺼내기
        
                        # 단일 데이터(1차원)인 경우, 강제로 배치 사이즈 1인 2차원으로 변환
                        if t.ndim == 1:
                            t = t.reshape(1, -1)
            
                        y, loss = self.get_loss(x, t) # 추론값, 오차 구하기
                        total_loss += float(loss)
                        self.backward(y, t) # 역전파
                        self.update() # 파라미터 갱신
        
                        # 값 확인
                        # if np.argmax(y) == np.argmax(t):
                        # 수정된 값 확인
                        pred = np.argmax(y, axis=1) # 100개 데이터 각각의 예측값 (100,)
                        truth = np.argmax(t, axis=1) # 100개 데이터 각각의 실제값 (100,)
                        correct += np.sum(pred == truth) # 맞춘 개수를 누적해서 더함
                            # correct += 1
                        total_samples += t.shape[0] # 방금 처리한 데이터 개수 누적
                    
                    # print문과 log 기록 수정 (self.x_train의 길이는 현재 600이므로 곱하기 배치사이즈 100을 해줍니다)
                    # total_samples = len(self.x_train) * len(x) # 600 * 100 = 60000
                    
                    print(f"Epoch {e+1} | Loss: {total_loss / len(self.x_train):.4f} | Accuracy: {correct / total_samples:.4f}")
                    self.loss_log.append(total_loss / len(self.x_train))
                    self.acc_log.append(correct / total_samples)