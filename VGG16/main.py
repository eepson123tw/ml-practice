import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 1. 載入預訓練的 VGG16 模型（包括頂層的全連接層）
model = VGG16(weights='imagenet')

# 2. 載入並預處理圖像
img_path = 'image.png'  # 替換為你的圖像路徑
img = image.load_img(img_path, target_size=(224, 224))  # VGG16 要求的輸入尺寸
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)  # 預處理步驟

# 3. 顯示預處理後的圖像（可選）
# 由於 VGG16 的預處理會改變圖像的色彩空間，這裡僅顯示原始圖像
plt.imshow(img)
plt.axis('off')
plt.show()

# 4. 進行預測
preds = model.predict(x)

# 5. 解析並顯示結果
print('預測結果：')
for pred in decode_predictions(preds, top=3)[0]:
    print(f"{pred[1]}: {pred[2]*100:.2f}%")
