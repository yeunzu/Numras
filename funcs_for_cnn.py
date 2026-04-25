import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 1. 패딩
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # 2. 결과 담을 빈 6차원
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # 3. 필터 크기만큼 for문
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_h
            # 넘파이 슬라이싱 이용 -> 한 번에 뭉탱이로 값 복사
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 4. 차원 순서 바꾸기 -> 2차원 평탄화
    # 축 순서: N, out_h, out_w, C, filter_h, filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col
    
    

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # 2차원 입력 -> 6차원으로 쪼개기
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    # 뭉탱이로 기울기 더하기
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 패딩이 있다면 패딩 자르고 원래 크기만 반한
    return img[:, :, pad:H + pad, pad:W + pad]
    