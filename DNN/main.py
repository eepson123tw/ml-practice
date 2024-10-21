import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
# 讀取 MNIST 資料集
with np.load('/Users/allen/Desktop/ml-practice/DNN/mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_valid, y_valid = f['x_test'], f['y_test']

print('Train Features Shape:', x_train.shape, 'labels Shape:', y_train.shape)
print('Valid Features Shape:', x_valid.shape, 'labels Shape:', y_valid.shape)

# 顯示第 0 筆資料
print('Label:', y_train[0])
plt.figure(figsize=(2, 2))
plt.imshow(x_train[0])
plt.show()

# 資料前處理（Data Preprocessing）
# 將 28 x 28 攤平為 784
reshape_train = x_train.reshape(x_train.shape[0], -1)
reshape_valid = x_valid.reshape(x_valid.shape[0], -1)

# 資料正規化（將 0~255 縮放為 0~1）
norm_train = reshape_train / 255.0
norm_valid = reshape_valid / 255.0

# One-hot encoding
onehot_train = to_categorical(y_train)
onehot_valid = to_categorical(y_valid)

print('Train Features Shape:', reshape_train.shape, 'labels Shape:', onehot_train.shape)
print('Valid Features Shape:', reshape_valid.shape, 'labels Shape:', onehot_valid.shape)

# 建立 DNN 模型
model = Sequential()
# 輸入層與第一個隱藏層
model.add(Dense(units=256, input_dim=784, activation='relu'))
# 第二個隱藏層
model.add(Dense(units=128, activation='relu'))
# 輸出層
model.add(Dense(units=10, activation='softmax'))  # 分類任務通常都使用 Softmax

# 建立損失函數與優化器
model.compile(loss='categorical_crossentropy',  # 分類任務選擇交叉熵作為 Loss function
              optimizer='adam',                 # 最常使用的優化器
              metrics=['accuracy'])             # 評估模型時的指標（這裡使用準確率）

# 開始訓練模型
history = model.fit(norm_train,          # 訓練資料
                    onehot_train,        # 訓練標籤
                    batch_size=128,      # 一次丟入多少訓練資料
                    epochs=10,           # 訓練次數
                    verbose=1,           # 顯示模式：0 = 不顯示，1 = 進度條，2 = 每個 epoch 一行
                    validation_data=(norm_valid, onehot_valid))  # 驗證數據集

# 儲存模型
# 只儲存權重
model.save_weights('model_weights.weights.h5')

# 儲存整個模型
model.save('model.h5')

# 如果使用 save_weights，需要重新創建模型結構
model = Sequential()
model.add(Dense(units=256, input_dim=784, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# 將訓練好的權重讀取回來
model.save_weights('model_weights.weights.h5')



# 使用load_model讀取模型
model = load_model('model.h5')
expand_data = np.expand_dims(norm_valid[0], axis = 0) 
model.predict(expand_data)
pred = np.argmax(model.predict(expand_data), axis = 1)
print('Label:', pred[0])
plt.figure(figsize=(2,2))
plt.imshow(x_valid[0])
plt.show()
