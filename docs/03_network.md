## Neurak Network model

### Network Overview

猶如前面提到的，模仿 OverFeat 我們可以用 Covolution 達到 sliding window 的效果。因此整個 network 必須具備以下的特性：
- 輸入：(m, 448, 448, 3)
- 輸出：(m, 7, 7, 2, 25) 也就是 (num_element, grid_size, grid_size, num_box, 5 + num_class)
- 輸出的所有元素都需要在 `[0, 1]` 之間 
- 輸出的最後一維的代表類別的 20 個元素總和為 1，因為代表機率

YOLO 在原始論文中建立的模型是 15 層 Conv layer 的架構，而在 YOLO2 中使用更深的模型並加入 Residual 的結構讓 back propagation 可以傳遞到底層（可以參考 [Deep Residual Network](https://arxiv.org/abs/1512.03385)）。不過太深的模型需要更多時間來訓練因此這裡我仿照論文建立 8 層的模型，如下：

![yolo_tiny_model](./images/yolo_tiny.png)

程式碼部分可以參考 `src/yolo_tiny.py`

- 整個模型的涵義是將原始圖片切割成 cell，之後對每個 cell 利用 sliding window（所有的 3x3 convolution），再藉著 sliding window 的輸出預測 bounding box 的位置和物體的類別（之後的兩個 1x1 convolution）。
- batch normalization 是為了讓輸出維持在 `[0, 1]` 之間
- dropout layer 產生 blending 的效果，減少 overfit 的影響
- 1x1 kernel 被稱為 network in network，效果猶如對每個 channel 使用 fully-connected layer
- 最後 activation 分別對 bounding box 項用 sigmoid 控制在 `[0, 1]`；對類別項用 softmax 讓總和為 1，最後再疊合
- 由於目前使用 Keras 建立模型，為了計算 loss 必須將最後結果疊合，如果直接用 tensorflow 可以省去這一步


### 損失函數(Loss)

損失函數的設計是 YOLO 重要的部分。除了考量到分類的準確度（可以用 cross-entry error）和 bounding box 的形狀差異（可能用 mean square error），也需要考慮到 confidence 項，如果一個 bounding box 的 confidence 接近 0，這樣子後面的幾項的誤差就不重要，也就是說以下兩個的誤差應該表示成很小：
```python
y_true = [   0 ,    0 ,   0,    0,    0,   0,   0, ... , 0]
y_pred = [0.01 , 0.99, 0.99, 0.99, 0.99, 0.9, 0.1, ... , 0]
```
反之以下的應該要表示很大
```python
y_true = [   1 ,  0.1,  0.2,  0.2,  0.2,   1,   0, ... , 0]
y_pred = [0.01 ,  0.1,  0.2,  0.2,  0.2,   1,   0, ... , 0]
```

因此論文給出的誤差函數如下：
![](./images/yolo_loss)

程式碼部分可以參考 `src/loss.py` 中的 `create_yolo_loss`


### Training and Result

目前訓練的方式是對每個 epoch 隨機對訓練資料作 shuffle，並以 batch size=32 為單位利用 adam 更新權重，執行 60 epochs 利用 GPU 花了兩小時後，可以得到約 0.49 的 train_loss，但在測試資料上 test_loss 約為 8.77。表示此模型確實可以逼近真正的 target function，但是有嚴重的 overfitting，未來必須針對這一點加上 Regularization 和使用 data argument 增加訓練資料。

[Back](../README.md)
