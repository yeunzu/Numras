from Model import MyModel
from layers import *
from activation_funcs import *
from error_funcs import *
from initializer import *
from optimizer import *
from funcs_for_cnn import *
from one_hot import *
import matplotlib.pyplot as plt
import numpy as np

print("loading data...")
mnist = np.load("mnist.npz")
x_train, y_train = mnist['x_train'], mnist['y_train']
x_test, y_test = mnist['x_test'], mnist['y_test']
print("data loading complete")

x_train = (x_train / 255.0).reshape(-1, 1, 28, 28)
x_test = (x_test / 255.0).reshape(-1, 1, 28, 28)

y_train = np.array([one_hot_encoding(i) for i in y_train])
# y_test = np.array([one_hot_encoding(i) for i in y_test])

cnn_model = MyModel()
cnn_model.load_data(data_kind='train', x=x_train, y=y_train)

cnn_model.add_layer(Conv(filter_num=10, filter_channels=1, filter_height=5, filter_width=5, stride=1, pad=0, learning_rate=0.01))
cnn_model.add_layer(Relu())
cnn_model.add_layer(Pooling(pool_h=2, pool_w=2, stride=2, pad=0))
cnn_model.add_layer(Flatten())
cnn_model.add_layer(Affine(input_size=1440, output_size=50, learning_rate=0.01))
cnn_model.add_layer(Relu())
cnn_model.add_layer(Affine(input_size=50, output_size=10, learning_rate=0.01))
cnn_model.add_layer(Softmax())

cnn_model.set_loss_func(func=CrossEntropyError())

cnn_model.data_batch_setting(setting_data='x_train', batch_size=100)
cnn_model.data_batch_setting(setting_data='y_train', batch_size=100)

cnn_model.learn(epoch=10)



print("start test")

predict_labels = []
batch_size = 100
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    pred_raw = cnn_model.predict(x_batch)
    pred_label = np.argmax(pred_raw, axis=1)
    predict_labels.extend(pred_label) # 예측된 100개 라벨을 리스트에 이어붙이기

predict_labels = np.array(predict_labels)
accuracy = np.sum(predict_labels == y_test) / len(y_test)
print(f"테스트 데이터 최종 정확도: {accuracy:.4f}")