import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import talib as ta

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
        
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()
        df['Volatility'] = df['log_returns'].rolling(window=30).std()
        df['RSI'] = ta.RSI(df['Close'])
        df['MACD'], df['Signal'], _ = ta.MACD(df['Close'])
        df['upper'], df['middle'], df['lower'] = ta.BBANDS(df['Close'])
        df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'])
        df['OBV'] = ta.OBV(df['Close'], df['Volume'])
        
        df = df.dropna()
        
        features = ['Close', 'Volume', 'log_returns', 'MA7', 'MA30', 'Volatility', 'RSI', 'MACD', 'Signal', 'upper', 'lower', 'ATR', 'OBV']
        
        dataset = df[features].values
        scaled_data = self.scaler.fit_transform(dataset)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # 0 index corresponds to 'Close' price
        
        return np.array(X), np.array(y), df

    def build_model(self, input_shape):
        input_layer = Input(shape=input_shape)
        
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)
        
        attention_out = Attention()([lstm_out, lstm_out])
        
        global_average = GlobalAveragePooling1D()(attention_out)
        
        dense_out = Dense(32, activation='relu')(global_average)
        output = Dense(1)(dense_out)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        return model

    def train_model(self, epochs=100, batch_size=32):
        X, y, _ = self.prepare_data()
        
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Load the best model
        self.model = load_model('best_model.h5')
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
            
            # Update the current batch for the next prediction
            new_row = np.zeros((1, current_batch.shape[2]))
            new_row[0, 0] = current_pred  # Assuming the first feature is the closing price
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1] = new_row
        
        future_prices = np.array(future_prices).reshape(-1, 1)
        
        # Inverse transform the predictions
        last_original_features = self.scaler.inverse_transform(X[-1][-1].reshape(1, -1))
        future_features = np.tile(last_original_features, (len(future_prices), 1))
        future_features[:, 0] = future_prices.flatten()
        
        adjusted_future_prices = self.scaler.inverse_transform(future_features)[:, 0]
        
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
        
        fig.add_trace(go.Scatter(x=df.index[-len(y):], y=actual_prices, name='Actual Price | 實際價格', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predicted_prices, name='Predicted Price | 預測價格', line=dict(color='red')), row=1, col=1)
        
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
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Historical Close | 歷史收盤價', line=dict(color='blue')), row=1, col=1)
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

st.sidebar.info("This app uses an advanced LSTM model with attention mechanism trained on historical data to predict cryptocurrency prices. Results are for reference only. | 本應用程序使用基於歷史數據訓練的高級LSTM模型和注意力機制來預測加密貨幣價格。結果僅供參考。")