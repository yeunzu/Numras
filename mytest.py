from Model import MyModel
from layers import *
from activation_funcs import *
from error_funcs import *
from initializer import *
from optimizer import *
import matplotlib.pyplot as plt
import numpy as np

print("loading data...")
mnist = np.load("mnist.npz")
x_train, y_train = mnist['x_train'], mnist['y_train']
x_test, y_test = mnist['x_test'], mnist['y_test']
print("data loading complete")


model = MyModel()
lr = 0.1
model.add_layer(Flatten())
model.add_layer(Affine(input_size=784, output_size=300, learning_rate=lr))
model.add_layer(Relu())
model.add_layer(Affine(input_size=300, output_size=100, learning_rate=lr))
model.add_layer(Relu())
model.add_layer(Affine(input_size=100, output_size=50, learning_rate=lr))
model.add_layer(Relu())
model.add_layer(Affine(input_size=50, output_size=10, learning_rate=lr))
model.add_layer(Softmax())

model.set_loss_func(func=CrossEntropyError())


def one_hot_encoding(label, max_num=10):
    tmp = [0 for i in range(max_num)]
    tmp[int(label)] = 1
    return tmp

def one_hot_decoding(label):
    return label.index(1)

    
x_train_ = x_train / 255.0 # 정규화
y_train_ = np.array([one_hot_encoding(data) for data in y_train]) # 원 핫 인코딩

model.load_data(data_kind='train', x=x_train_, y=y_train_)
model.data_batch_setting(setting_data='x_train', batch_size=100)
model.data_batch_setting(setting_data='y_train', batch_size=100)

print("start learning")
epoch = 10
model.learn(epoch=epoch)