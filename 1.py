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
import tensorflow as tf

class EnhancedCryptoAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        
    @st.cache_data(ttl=3600)
    def fetch_data(self, years=5):
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=years)
        self.data = yf.download(self.ticker, start=start_date, end=end_date)
        return self.data

    def prepare_data(self, sequence_length=60):
        df = self.data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['Volatility'] = df['Returns'].rolling(window=30).std()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal'] = self.calculate_macd(df['Close'])
        
        df = df.dropna()
        
        features = ['Close', 'Volume', 'Returns', 'MA7', 'MA30', 'Volatility', 'RSI', 'MACD', 'Signal']
        
        dataset = df[features].values
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # 0 index corresponds to 'Close' price
        
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
        return macd, signal_line

    @st.cache_resource
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        X, y, _ = self.prepare_data()
        
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        return history

    def predict_future_price(self, days=30):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        X, _, df = self.prepare_data()
        last_sequence = X[-1]
        current_batch = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
        
        future_prices = []
        
        for _ in range(days):
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            future_prices.append(current_pred)
            
            new_row = np.zeros((1, current_batch.shape[2]))
            new_row[0, 0] = current_pred
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1] = new_row
        
        future_prices = np.array(future_prices).reshape(-1, 1)
        
        last_original_price = df['Close'].iloc[-1]
        first_predicted_price = self.scaler.inverse_transform(future_prices)[0][0]
        scaling_factor = last_original_price / first_predicted_price
        
        adjusted_future_prices = self.scaler.inverse_transform(future_prices).flatten() * scaling_factor
        
        return adjusted_future_prices

    def visualize_model_performance(self):
        X, y, df = self.prepare_data()
        predictions = self.model.predict(X, verbose=0).flatten()
        
        actual_prices = self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        predicted_prices = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        mse = np.mean((actual_prices - predicted_prices)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=(f'{self.ticker} Model Performance | 模型性能', 'Prediction Error | 預測誤差'))
        
        fig.add_trace(go.Scatter(x=df.index[-len(y):], y=actual_prices, name='Actual Price | 實際價格'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predicted_prices, name='Predicted Price | 預測價格'), row=1, col=1)
        
        errors = actual_prices - predicted_prices
        fig.add_trace(go.Scatter(x=df.index[-len(errors):], y=errors, name='Prediction Error | 預測誤差'), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=800, title_text=f'{self.ticker} Model Performance Analysis | 模型性能分析 (RMSE: {rmse:.2f}, MAE: {mae:.2f})', showlegend=True)
        return fig

@st.cache_resource
def get_analyzer(ticker):
    return EnhancedCryptoAnalyzer(ticker)

st.set_page_config(page_title="Crypto Analyzer | 加密貨幣分析器", page_icon="📊", layout="wide")

st.title("Cryptocurrency Analysis and Prediction | 加密貨幣分析與預測")

st.sidebar.header("Settings | 設置")
ticker = st.sidebar.selectbox("Select Cryptocurrency | 選擇加密貨幣", ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "DOT-USD"])
years = st.sidebar.slider("Years of Historical Data | 歷史數據年數", 1, 10, 5)
future_days = st.sidebar.slider("Days to Predict | 預測天數", 7, 90, 30)

analyzer = get_analyzer(ticker)

if st.sidebar.button("Start Analysis | 開始分析"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress):
        progress_bar.progress(progress)
        status_text.text(f"Progress | 進度: {progress}%")

    with st.spinner("Fetching and processing data... | 獲取和處理數據中..."):
        df = analyzer.fetch_data(years=years)
        update_progress(20)

        status_text.text("Training model... | 訓練模型中...")
        history = analyzer.train_model(epochs=100)
        update_progress(60)
        
        status_text.text("Predicting future prices... | 預測未來價格中...")
        future_prices = analyzer.predict_future_price(days=future_days)
        update_progress(80)

        status_text.text("Creating charts... | 創建圖表中...")
        _, _, df = analyzer.prepare_data()
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Price Prediction | 價格預測", "Trading Volume | 交易量", "RSI"))

        # Price Prediction | 價格預測
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Close | 歷史收盤價'), row=1, col=1)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Predicted Price | 預測價格', line=dict(color='red', dash='dash')), row=1, col=1)
        
        # Trading Volume | 交易量
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Trading Volume | 交易量'), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=900, title_text=f"{ticker} Comprehensive Analysis | 綜合分析", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.header("Price Prediction | 價格預測")
        st.write(f"Predicted price after {future_days} days | {future_days}天後的預測價格: ${future_prices[-1]:.2f}")

        st.header("Model Performance | 模型性能")
        performance_fig = analyzer.visualize_model_performance()
        st.plotly_chart(performance_fig, use_container_width=True)

        update_progress(100)
        status_text.text("Analysis complete | 分析完成")

st.sidebar.info("This app uses an LSTM model trained on historical data to predict cryptocurrency prices. Results are for reference only. | 本應用程序使用基於歷史數據訓練的LSTM模型來預測加密貨幣價格。結果僅供參考。")