# 關於CSIC數據集的訓練

## 數據集說明
使用SVM（支持向量機）模型，對CSIC數據集進行訓練

本次訓練用到的數據集為CSIC2010的測試集（normal.txt & anomalous.txt）

因為SVM是一種貳分類的方法，訓練階段需要兩種類型的數據

而CSIC2010本身提供的訓練集只包含normal

故使用CSIC2010提供的測試集，進一步將其劃分為訓練集和測試集

## 程式流程
1. 首先對文本進行簡單的預處理（提取文本中的URL）
2. 然後對文本進行項量化，本實驗採用TF-IDF來向量化文本
3. 使用sklearn提供的SVM進行訓練
4. 對模型進行測試，最終score在0.9左右

## 說明
本實驗僅作為參考。
