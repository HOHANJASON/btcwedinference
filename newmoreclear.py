import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class EnhancedCryptoAnalyzer:
    def __init__(self, ticker, interval='15m'):
        self.ticker = ticker
        self.interval = interval
        self.data = None
        self.intraday_data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.close_scaler = MinMaxScaler()


    def fetch_data(self, days=30):
        crypto = yf.Ticker(self.ticker)
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        
        self.data = crypto.history(start=start_date, end=end_date, interval=self.interval)
        
        while len(self.data) < days * 24 * 4 and start_date > end_date - pd.Timedelta(days=60):
            start_date = start_date - pd.Timedelta(days=5)
            temp_data = crypto.history(start=start_date, end=end_date, interval=self.interval)
            self.data = pd.concat([temp_data, self.data]).drop_duplicates()

        self.data.sort_index(inplace=True)
        
        self.intraday_data = crypto.history(start=start_date, end=end_date, interval='4h')

    def prepare_data(self):
        df = self.data.copy()
        
        df['Returns'] = df['Close'].pct_change()
        df['MA7'] = df['Close'].rolling(window=7*4).mean()
        df['MA30'] = df['Close'].rolling(window=30*4).mean()
        
        rsi_indicator = RSIIndicator(df['Close'])
        df['RSI'] = rsi_indicator.rsi()
        
        macd_indicator = MACD(df['Close'])
        df['MACD'] = macd_indicator.macd()
        df['Signal'] = macd_indicator.macd_signal()
        
        bb_indicator = BollingerBands(df['Close'])
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        
        df['Support'] = df['Close'].rolling(window=20).min()
        
        df['Hour'] = df.index.hour / 24
        df['DayOfWeek'] = df.index.dayofweek / 7
        
        df.dropna(inplace=True)
        
        features = ['Close', 'Volume', 'Returns', 'MA7', 'MA30', 'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower', 'Hour', 'DayOfWeek']
        X = df[features].values
        y = df['Close'].values
        
        X = self.scaler.fit_transform(X)
        y = self.close_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        seq_length = 20
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq), df #標
    
    def visualize_data_and_prediction(self, future_prices):
        _, _, df = self.prepare_data()
        fng_data = self.get_fear_and_greed_index()
        
        fig = make_subplots(rows=6, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=(f'{self.ticker} 價格預測', '每日蠟燭圖', '4小時蠟燭圖', '交易量', 'RSI & MACD', '恐懼&貪婪指數'))
        
        # 價
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='歷史收盤價'), row=1, col=1)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_prices)+1, freq='15T')[1:]
        all_prices = np.concatenate([df['Close'].values[-1:], future_prices])
        all_dates = pd.concat([df.index[-1:], future_dates])
        fig.add_trace(go.Scatter(x=all_dates, y=all_prices, name='預測價格', line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='7日移動平均線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], name='30日移動平均線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='支撐位', line=dict(color='green', dash='dot')), row=1, col=1)
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

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
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
        return model

    def train_model(self, epochs=100, batch_size=32):
        X, y, _ = self.prepare_data()
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(factor=0.2, patience=10, min_lr=0.0001)
            
            self.model.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=batch_size, 
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )

    def predict_future_price(self, periods=96):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train_model() first.")
        
        last_sequence = self.prepare_data()[0][-1]
        current_batch = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
        
        future_prices = []
        
        for i in range(periods):
            current_pred = self.model.predict(current_batch, verbose=0)[0]
            future_prices.append(current_pred)
            
            new_row = current_batch[0, -1, :].copy()
            new_row[0] = current_pred
            new_row[2] = (current_pred - current_batch[0, -1, 0]) / current_batch[0, -1, 0]
            new_row[3] = np.mean(current_batch[0, -28:, 0])
            new_row[4] = np.mean(current_batch[0, -120:, 0])
            new_row[9] = (i % 24) / 24
            new_row[10] = ((current_batch[0, -1, 10] * 7 + (i // 24)) % 7) / 7
            
            current_batch = np.roll(current_batch, -1, axis=1)
            current_batch[0, -1] = new_row
        
        future_prices = np.array(future_prices).reshape(-1, 1)
        
        return self.close_scaler.inverse_transform(future_prices).flatten()

    def get_fear_and_greed_index(self):
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url)
        data = response.json()
        return pd.DataFrame(data['data'])

    def visualize_data_and_prediction(self, future_prices):
        _, _, df = self.prepare_data()
        fng_data = self.get_fear_and_greed_index()
        
        fig = make_subplots(rows=6, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05, 
                            subplot_titles=(f'{self.ticker} 價格預測', '每日蠟燭圖', '4小時蠟燭圖', '交易量', 'RSI & MACD', '恐懼&貪婪指數'))
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='歷史收盤價'), row=1, col=1)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_prices)+1, freq='15T')[1:]
        all_prices = np.concatenate([df['Close'].values[-1:], future_prices])
        all_dates = pd.to_datetime(np.concatenate([df.index[-1:].values, future_dates.values]))
        fig.add_trace(go.Scatter(x=all_dates, y=all_prices, name='預測價格', line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='7日移動平均線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA30'], name='30日移動平均線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support'], name='支撐位', line=dict(color='green', dash='dot')), row=1, col=1) # forecast

        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='每日蠟燭圖'), row=2, col=1)

        fig.add_trace(go.Candlestick(x=self.intraday_data.index,
                                     open=self.intraday_data['Open'],
                                     high=self.intraday_data['High'],
                                     low=self.intraday_data['Low'],
                                     close=self.intraday_data['Close'],
                                     name='4小時蠟燭圖'), row=3, col=1) #???

        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='交易量'), row=4, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'), row=5, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal Line'), row=5, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=5, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=5, col=1) # RSI y MACD
        
        fig.add_trace(go.Scatter(x=pd.to_datetime(fng_data['timestamp'], unit='s'), y=fng_data['value'].astype(float), name='恐懼&貪婪指數'), row=6, col=1)
        
        fig.update_layout(height=1800, title_text=f'{self.ticker} 綜合分析', showlegend=False)
        fig.update_xaxes(rangeslider_visible=False)
        fig.show()

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
        fig.show()

# pyt
ticker = 'BTC-USD'
analyzer = EnhancedCryptoAnalyzer(ticker, interval='15m')

print(f"正在獲取 {ticker} 的歷史數據...")
analyzer.fetch_data(days=30)

print("正在訓練模型，這可能需要一些時間...")
analyzer.train_model(epochs=100)

print("預測24小時的價格")
future_prices = analyzer.predict_future_price(periods=96)

print("生成綜合分析圖表...")
analyzer.visualize_data_and_prediction(future_prices)

print("生成模型性能圖表...")
analyzer.visualize_model_performance()

print(f"\n{ticker} 24小时后的预测价格: ${future_prices[-1]:.2f}")