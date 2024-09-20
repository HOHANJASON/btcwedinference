# Cryptocurrency Analysis and Prediction
基於模仿漲幅取向而製作的
# 連結
[Cryptocurrency Analysis and Prediction | 加密貨幣分析與預測](https://btcwedinference-byhohanjason.streamlit.app/)

1.基本概述
- 以streamlit為網路應用為基礎 
- 有基礎介面能使用但目前還能在更人性化一些 之後應該會改成html 充當UI
- 提供靜態方法計算 RSI、MACD 和支撐位 使用 Plotly 繪製圖表
- 

2.主要特點
- EnhancedCryptoAnalyzer 以此類進行資料擷取
- prepare_data 方法計算收益率、移動平均線、波動率、相對強弱指數（RSI）、MACD指標和支撐位
- fetch_data 引用 yfinance 從中抓取歷史數據
- predict_future_price 方法用來預測未來指定天數的價格 但相對的放大天數導致必須使用更大量的資料
- visualize_model_performance 顯示模型的性能 計算預測的均方根誤差（RMSE）和平均絕對誤差（MAE）

3.瓶頸與突破
- 目前因為資料量含時間變量問題 所以推測時間有一定的範圍問題
- 還有出自LSTM模型的訓練需要較高的計算 非常耗時
- LSTM的模型架構是固定的 有明顯限制
- Plotly的可視化太簡單了 


