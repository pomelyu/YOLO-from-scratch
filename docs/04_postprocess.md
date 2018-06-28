## 後處理

經過類神經網路的運算之後，會產生許多候選的 bounding box，後處理所要作的事情就是進一步的篩選這些 bounding box，主要分為兩個部分
- Thresholding：去除 confience 太低的 bounding box，並取得物件的類別
- Non-maximum supression：去除太過密集的 bounding box。這個步驟在 R-CNN 中也有採用，雖然已經跳出處理過程中 end-to-end 的範疇，但是 YOLO 作者認為這一步對於整個速度的影響不大
這部份是 Andrew Ng 課程中作業的部分，因此可以參考相關的說明

### Thresholding
1. 將類別的機率乘上 confience
2. 選出機率最大的類別，如果該機率（稱為 score）大於某個 threshold，認定 bounding box 屬於該類別，否則捨棄這個 bounding box
3. 回傳剩餘的 bounding box 

### Non-maximum supression
1. 依照 score 的高低，排列 bounding box
2. 依序選取最大的 box，將和它重疊的 box 刪除。是否重疊可以用 IOU（intersection over union）的大小判斷。
3. 只到所有的 box 都被選取或是選取的數目到達預設值

tensorflow 中提供 `tf.image.non_max_suppression` 可以進行這個步驟，不過需要注意的是輸入的 box 座標是對角線的`(x1, y1, x2, y2)`，因此需要把模型出來的結果作座標轉換。

[Back](../README.md)
