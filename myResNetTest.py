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

resnet_model = MyModel()
resnet_model.add_layer(Conv(filter_num=10, filter_channels=1, filter_height=3, filter_width=3))
resnet_model.add_layer(Relu())
resnet_model.add_layer(ResidualBlock(10, 3, 3))
resnet_model.add_layer(ResidualBlock(10, 3, 3))
resnet_model.add_layer(Pooling(pool_h=2, pool_w=2, stride=2, pad=0))
resnet_model.add_layer(Flatten())
resnet_model.add_layer(Affine(input_size=1690, output_size=10)) # 사이즈 연산 과정은 아래 주석 참고
resnet_model.add_layer(Softmax())

"""
사이즈 연산
1. 입력 : (Batch, 1, 28, 28)
2. 첫 Conv: 필터 10, 크기 3x3, 패딩 0
    크기 계산: (28 + 0 - 3)/1 + 1 = 26
    출력 : (Batch, 10, 26, 26)
3. ResidualBlock 1, 2: 출력 유지 (Batch, 10, 26, 26)
4. Pooling: 2x2 풀링 계층 -> 크기 절반으로
    출력 : (Batch, 10, 13, 13)
5. Flatten: 10 x 13 x 13 = 1690
"""

resnet_model.set_loss_func(func=CrossEntropyError())
resnet_model.load_data(data_kind='train', x=x_train, y=y_train)

resnet_model.data_batch_setting(setting_data='x_train', batch_size=100)
resnet_model.data_batch_setting(setting_data='y_train', batch_size=100)
resnet_model.learn(epoch=10)

print("start test")
predict_labels = []
batch_size = 100
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    pred_raw = resnet_model.predict(x_batch)
    pred_label = np.argmax(pred_raw, axis=1)
    predict_labels.extend(pred_label) # 예측된 100개 라벨을 리스트에 이어붙이기

predict_labels = np.array(predict_labels)
accuracy = np.sum(predict_labels == y_test) / len(y_test)
print(f"테스트 데이터 최종 정확도: {accuracy:.4f}")