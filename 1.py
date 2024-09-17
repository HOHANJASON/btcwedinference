import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests

class EnhancedCryptoAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.intraday_data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.close_scaler = MinMaxScaler() #首
        
    def fetch_data(self, years=10):
        crypto = yf.Ticker(self.ticker)
        self.data = crypto.history(period=f"{years}y", interval="1d")
        
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=7)
        self.intraday_data = crypto.history(start=start_date, end=end_date, interval="4h")

    def prepare_data(self, sequence_length=60):
        df = self.data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['Volatility'] = df['Returns'].rolling(window=30).std()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = self.calculate_macd(df['Close'])
        df['Support'] = self.calculate_support(df['Close'])  #缺少附加偵查項 代檢
        
        df = df.dropna()
        
        features = ['Close', 'Volume', 'Returns', 'MA7', 'MA30', 'Volatility', 'RSI', 'MACD', 'Signal', 'Support']
        target = 'Close'
        
        dataset = df[features].values
        scaled_data = self.scaler.fit_transform(dataset)
        
        self.close_scaler.fit(df[['Close']])
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, features.index(target)])
        
        return np.array(X), np.array(y), df

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line #4/sep/24

    @staticmethod
    def calculate_support(prices, window=20):
        return prices.rolling(window=window).min()

    def get_fear_and_greed_index(self):
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url)
        data = response.json()
        return pd.DataFrame(data['data'])

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train_model(self, epochs=100, batch_size=32):
        X, y, _ = self.prepare_data()
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5)
            
            self.model.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )

    def predict_future_price(self, days=30):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        last_sequence = self.prepare_data()[0][-1]
        current_batch = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
        
        future_prices = []
        
        for _ in range(days):
            current_pred = self.model.predict(current_batch)[0]
            future_prices.append(current_pred)
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1, 0] = current_pred
        
        future_prices = np.array(future_prices).reshape(-1, 1)
        
        return self.close_scaler.inverse_transform(future_prices).flatten()

    def visualize_model_performance(self):
        X, y, df = self.prepare_data()
        predictions = self.model.predict(X).flatten()
        
        actual_prices = self.close_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        predicted_prices = self.close_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        mse = np.mean((actual_prices - predicted_prices)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=(f'{self.ticker} 模型性能', '預測誤差'))
        
        fig.add_trace(go.Scatter(x=df.index[-len(y):], y=actual_prices, name='實際價格'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predicted_prices, name='預測價格'), row=1, col=1)
        
        errors = actual_prices - predicted_prices
        fig.add_trace(go.Scatter(x=df.index[-len(errors):], y=errors, name='預測誤差'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=800, title_text=f'{self.ticker} 模型性能分析 (RMSE: {rmse:.2f}, MAE: {mae:.2f})', showlegend=True)
        return fig #new build

st.set_page_config(page_title="加密貨幣分析器", page_icon="📊", layout="wide")

st.title("加密貨幣分析與預測")

st.sidebar.header("設置")
ticker = st.sidebar.selectbox("選擇加密貨幣", ["BTC-USD", "ETH-USD"])
years = st.sidebar.slider("歷史數據年數", 1, 10, 5)
future_days = st.sidebar.slider("預測天數", 7, 90, 30)

st.sidebar.header("選擇要顯示的圖表")
show_price_prediction = st.sidebar.checkbox("價格預測", value=True)
show_price_ma = st.sidebar.checkbox("價格與移動平均線", value=True)
show_volume = st.sidebar.checkbox("交易量", value=True)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_fear_greed = st.sidebar.checkbox("恐懼&貪婪指數", value=True)

@st.cache_resource
def get_analyzer(ticker, years):
    analyzer = EnhancedCryptoAnalyzer(ticker)
    analyzer.fetch_data(years=years)
    return analyzer

analyzer = get_analyzer(ticker, years)

if st.sidebar.button("開始分析"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress):
        progress_bar.progress(progress)
        status_text.text(f"進度: {progress}%")

    status_text.text("正在訓練模型...")
    update_progress(10)
    analyzer.train_model(epochs=100)
    update_progress(50)
    
    status_text.text("正在預測未來價格...")
    future_prices = analyzer.predict_future_price(days=future_days)
    update_progress(70)

    status_text.text("正在準備數據...")
    _, _, df = analyzer.prepare_data()
    update_progress(80)

    status_text.text("正在獲取恐懼&貪婪指數...")
    fng_data = analyzer.get_fear_and_greed_index()
    update_progress(90)

    status_text.text("正在創建圖表...")
    fig = make_subplots(rows=sum([show_price_prediction, show_price_ma, show_volume, show_rsi, show_macd, show_fear_greed]), 
                        cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    row = 1

    if show_price_prediction:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='歷史收盤價'), row=row, col=1)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='預測價格', line=dict(color='red', dash='dash')), row=row, col=1)
        fig.update_yaxes(title_text="價格", row=row, col=1)
        row += 1

    if show_price_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='7日MA'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], name='30日MA'), row=row, col=1)
        fig.update_yaxes(title_text="價格", row=row, col=1)
        row += 1

    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='交易量'), row=row, col=1)
        fig.update_yaxes(title_text="交易量", row=row, col=1)
        row += 1

    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
        fig.update_yaxes(title_text="RSI", row=row, col=1)
        row += 1

    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal'), row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1)
        row += 1

    if show_fear_greed:
        fig.add_trace(go.Scatter(x=pd.to_datetime(fng_data['timestamp'], unit='s'), 
                                 y=fng_data['value'].astype(float), name='F&G Index'), row=row, col=1)
        fig.update_yaxes(title_text="恐懼&貪婪指數", row=row, col=1)

    fig.update_layout(height=300*row, title_text=f"{ticker} 綜合分析", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.header("價格預測")
    st.write(f"{future_days}天後的預測價格: ${future_prices[-1]:.2f}")

    st.header("模型性能")
    performance_fig = analyzer.visualize_model_performance()
    st.plotly_chart(performance_fig, use_container_width=True)

    update_progress(100)
    status_text.text("分析完成")

st.sidebar.info("自歷史數據訓練LSTM模型來預測加密貨幣價格。輸錢別怪我結果僅供參考。")

#代碼我開源了 你們可以研究遺下優化方式 我現階段在想有沒有甚麼算法能平替LSTM 這模型速度太慢了
