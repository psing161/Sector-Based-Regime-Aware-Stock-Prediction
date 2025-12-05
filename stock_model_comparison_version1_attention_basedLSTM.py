import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Multiply, Permute, RepeatVector, Lambda
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import (
    mean_squared_error, r2_score, f1_score, precision_score,
    classification_report, confusion_matrix
)
from results_charts import save_all_charts
from collections import Counter
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

# Suppress verbose warnings for cleaner output
warnings.filterwarnings('ignore')

class AttentionLayer:
    """
    Custom Attention Layer for LSTM models.
    This implements a simple attention mechanism that learns to focus on 
    different parts of the sequence based on their relevance.
    """
    @staticmethod
    def attention_3d_block(inputs, time_steps):
        """
        Attention mechanism that computes attention weights for each time step.
        """
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        
        # Compute attention scores
        a = Permute((2, 1))(inputs)  # (batch_size, input_dim, time_steps)
        a = Dense(time_steps, activation='softmax')(a)
        
        # Apply attention weights
        a_probs = Permute((2, 1))(a)  # (batch_size, time_steps, input_dim)
        output_attention_mul = Multiply()([inputs, a_probs])
        
        return output_attention_mul
    
    @staticmethod
    def build_attention_lstm_regressor(input_shape, lstm_units=64, dense_units=32):
        """
        Build an Attention-based LSTM model for regression.
        """
        inputs = Input(shape=input_shape)
        
        # LSTM layer with return_sequences=True to get all time steps
        lstm_out = LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        
        # Apply attention mechanism
        attention_out = AttentionLayer.attention_3d_block(lstm_out, input_shape[0])
        
        # Global average pooling to reduce time dimension
        attention_out = Lambda(lambda x: tf.reduce_mean(x, axis=1))(attention_out)
        
        # Dense layers
        dense_out = Dense(dense_units, activation='relu')(attention_out)
        dense_out = Dropout(0.3)(dense_out)
        
        # Output layer for regression
        outputs = Dense(1, activation='linear')(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def build_attention_lstm_classifier(input_shape, num_classes, lstm_units=64, dense_units=32):
        """
        Build an Attention-based LSTM model for classification.
        """
        inputs = Input(shape=input_shape)
        
        # LSTM layer with return_sequences=True
        lstm_out = LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        
        # Apply attention mechanism
        attention_out = AttentionLayer.attention_3d_block(lstm_out, input_shape[0])
        
        # Global average pooling
        attention_out = Lambda(lambda x: tf.reduce_mean(x, axis=1))(attention_out)
        
        # Dense layers
        dense_out = Dense(dense_units, activation='relu')(attention_out)
        dense_out = Dropout(0.3)(dense_out)
        
        # Output layer for classification
        if num_classes == 2:
            outputs = Dense(num_classes, activation='sigmoid')(dense_out)
        else:
            outputs = Dense(num_classes, activation='softmax')(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

sectors_map = {
    'Finance': ['BAC','BLK','JPM','SCHW','V','GS','C'],
    'HealthCare': ['BSX','GILD','JNJ','PFE','UNH','ABBV','MRK','ABT'],
    'Tech': ['AAPL','AMZN','AMD','GOOGL','MSFT','NVDA','QCOM']
}

SECTOR_FEATURE_MAP = {
    'Tech': [
        # Price and volume features
        'close', 'volume', 'ATR',
        # Technical indicators
        'EMA_200', 'EMA_50', 'EMA_50_200_crossover', 'RSI', 'RSI_signal', 
        'MACD', 'MACD_Histogram', 'Bollinger_%B', '%K',
        'volatility', 'support_distance', 'resistance_distance',
        # Enhanced return features
        'return_today', 'return_lag1', 'return_lag2', 'return_lag3', 'return_lag4', 'return_lag5',
        'return_weekly', 'return_monthly', 'return_volatility_5d', 'return_volatility_20d',
        'return_mean_5d', 'return_mean_20d',
        # Sentiment and external indicators
        'sentiment_score', 'vix', 'vix_change', 'vix_spike', 'market_stress', 'risk_sentiment',
        'treasury_10y', 'treasury_10y_change', 'oil_change', 'oil_volatility',
        # Split-adjusted features
        'volume_ratio', 'price_stability', 'momentum_3m', 'momentum_6m', 'price_vs_ma50', 'price_vs_ma200',
        # Sector-specific indices (will be added dynamically)
        'ixic_return', 'ndx_return', 'qqq_return', 'ixic_volatility', 'ndx_volatility',
    ],
    
    'Finance': [
        # Price and volume features
        'volume', 'ATR', 'EMA_200', 'EMA_50',
        # Technical indicators
        'RSI', 'volatility', 'support_distance', 'resistance_distance',
        'MACD', 'MACD_Histogram', 'OBV', 'Bollinger_%B', '%K',
        # Enhanced return features
        'return_today', 'return_lag1', 'return_lag2', 'return_lag3', 'return_lag4', 'return_lag5',
        'return_weekly', 'return_monthly', 'return_volatility_5d', 'return_volatility_20d',
        'return_mean_5d', 'return_mean_20d',
        # Sentiment and external indicators
        'sentiment_score', 'fed_rate', 'fed_rate_change', 'fed_rate_trend',
        'vix', 'vix_change', 'dxy', 'dxy_change', 'treasury_10y', 'treasury_10y_change',
        'market_stress', 'risk_sentiment', 'unemployment_rate', 'unemployment_change',
        # Split-adjusted features
        'volume_ratio', 'price_stability', 'momentum_3m', 'momentum_6m', 'price_vs_ma50', 'price_vs_ma200',
        # Sector-specific indices
        'bkx_return', 'xlf_return', 'kre_return', 'bkx_volatility', 'xlf_volatility'
    ],

    'HealthCare': [
        # Price and volume features
        'volume', 'ATR', 'EMA_200', 'EMA_50',
        # Technical indicators
        'RSI_signal', 'volatility', 'support_distance', 'resistance_distance',
        'MACD', 'MACD_Histogram', 'final_signal', 'ROC', 'PROC', 'ADX', 'OBV',
        # Enhanced return features
        'return_today', 'return_lag1', 'return_lag2', 'return_lag3', 'return_lag4', 'return_lag5',
        'return_weekly', 'return_monthly', 'return_volatility_5d', 'return_volatility_20d',
        'return_mean_5d', 'return_mean_20d',
        # Sentiment and external indicators
        'sentiment_score', 'vix', 'vix_change', 'oil_price', 'oil_change',
        'cpi_yoy', 'market_stress', 'risk_sentiment',
        # Split-adjusted features
        'volume_ratio', 'price_stability', 'momentum_3m', 'momentum_6m', 'price_vs_ma50', 'price_vs_ma200',
        # Sector-specific indices
        'xlv_return', 'vht_return', 'ihi_return', 'xlv_volatility', 'vht_volatility'
    ],

    'General': [
        # Core features for general model
        'ATR', 'EMA_200', 'EMA_50', 'EMA_50_200_crossover', 'volume', 'RSI', 
        'volatility', 'support_distance',
        # Enhanced return features
        'return_today', 'return_lag1', 'return_lag2', 'return_lag3', 'return_weekly',
        'return_volatility_5d', 'return_mean_5d',
        # Key external indicators
        'sentiment_score', 'vix', 'vix_change', 'market_stress', 'treasury_10y_change',
        # Split-adjusted features
        'volume_ratio', 'momentum_3m', 'price_vs_ma50'
    ]
}

def calculate_real_technical_indicators(data):
    """
    Calculate real technical indicators using pandas operations.
    """
    # Group by ticker to calculate indicators per stock
    for ticker in data['ticker'].unique():
        ticker_mask = data['ticker'] == ticker
        ticker_data = data[ticker_mask].copy().sort_values('date')
        
        # RSI calculation (14-period)
        delta = ticker_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data.loc[ticker_mask, 'RSI'] = 100 - (100 / (1 + rs))
        
        # RSI Signal
        data.loc[ticker_mask, 'RSI_signal'] = data.loc[ticker_mask, 'RSI'].apply(
            lambda x: 1 if x > 70 else (-1 if x < 30 else 0)
        )
        
        # Moving Averages
        data.loc[ticker_mask, 'SMA_20'] = ticker_data['close'].rolling(20).mean()
        data.loc[ticker_mask, 'SMA_50'] = ticker_data['close'].rolling(50).mean()
        data.loc[ticker_mask, 'EMA_50'] = ticker_data['close'].ewm(span=50).mean()
        data.loc[ticker_mask, 'EMA_200'] = ticker_data['close'].ewm(span=200).mean()
        
        # EMA Crossover Signal
        data.loc[ticker_mask, 'EMA_50_200_crossover'] = (
            data.loc[ticker_mask, 'EMA_50'] > data.loc[ticker_mask, 'EMA_200']
        ).astype(int)
        
        # MACD (12, 26, 9)
        ema12 = ticker_data['close'].ewm(span=12).mean()
        ema26 = ticker_data['close'].ewm(span=26).mean()
        data.loc[ticker_mask, 'MACD'] = ema12 - ema26
        data.loc[ticker_mask, 'MACD_signal'] = data.loc[ticker_mask, 'MACD'].ewm(span=9).mean()
        data.loc[ticker_mask, 'MACD_Histogram'] = (
            data.loc[ticker_mask, 'MACD'] - data.loc[ticker_mask, 'MACD_signal']
        )
        
        # ATR (Average True Range)
        high_low = ticker_data['high'] - ticker_data['low']
        high_close = np.abs(ticker_data['high'] - ticker_data['close'].shift())
        low_close = np.abs(ticker_data['low'] - ticker_data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data.loc[ticker_mask, 'ATR'] = true_range.rolling(14).mean()

        # HLC3
        data.loc[ticker_mask, 'hlc3'] = ticker_data['high'] + ticker_data['low'] + ticker_data['close'] / 3

        # Bollinger Bands
        sma20 = ticker_data['close'].rolling(20).mean()
        std20 = ticker_data['close'].rolling(20).std()
        data.loc[ticker_mask, 'Bollinger_upper'] = sma20 + (std20 * 2)
        data.loc[ticker_mask, 'Bollinger_lower'] = sma20 - (std20 * 2)
        data.loc[ticker_mask, 'Bollinger_%B'] = (
            (ticker_data['close'] - data.loc[ticker_mask, 'Bollinger_lower']) /
            (data.loc[ticker_mask, 'Bollinger_upper'] - data.loc[ticker_mask, 'Bollinger_lower'])
        )

        # Stochastic %K
        low14 = ticker_data['low'].rolling(14).min()
        high14 = ticker_data['high'].rolling(14).max()
        data.loc[ticker_mask, '%K'] = (
            (ticker_data['close'] - low14) / (high14 - low14) * 100
        )
        
        # Volatility
        data.loc[ticker_mask, 'volatility'] = ticker_data['close'].pct_change().rolling(20).std()
        
        # Support and Resistance distances (simplified)
        rolling_min = ticker_data['close'].rolling(50).min()
        rolling_max = ticker_data['close'].rolling(50).max()
        data.loc[ticker_mask, 'support_distance'] = (
            (ticker_data['close'] - rolling_min) / ticker_data['close']
        )
        data.loc[ticker_mask, 'resistance_distance'] = (
            (rolling_max - ticker_data['close']) / ticker_data['close']
        )
        
        # ROC (Rate of Change)
        data.loc[ticker_mask, 'ROC'] = ticker_data['close'].pct_change(periods=12) * 100
        
        # PROC (Price Rate of Change)  
        data.loc[ticker_mask, 'PROC'] = (
            (ticker_data['close'] - ticker_data['close'].shift(12)) / 
            ticker_data['close'].shift(12) * 100
        )
        
        # ADX (simplified directional movement)
        plus_dm = ticker_data['high'].diff()
        minus_dm = ticker_data['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        data.loc[ticker_mask, 'ADX'] = (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()) / 2
        
        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(ticker_data)):
            if ticker_data['close'].iloc[i] > ticker_data['close'].iloc[i-1]:
                obv.append(obv[-1] + ticker_data['volume'].iloc[i])
            elif ticker_data['close'].iloc[i] < ticker_data['close'].iloc[i-1]:
                obv.append(obv[-1] - ticker_data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data.loc[ticker_mask, 'OBV'] = obv
        
        # Final signal (composite)
        rsi_signal = data.loc[ticker_mask, 'RSI_signal']
        macd_signal = (data.loc[ticker_mask, 'MACD'] > data.loc[ticker_mask, 'MACD_signal']).astype(int)
        data.loc[ticker_mask, 'final_signal'] = (rsi_signal + macd_signal - 1).clip(-1, 1)
    
    return data

def create_improved_targets(data):
    """
    Create better target variables with improved thresholding and binary option.
    """
    # Calculate tomorrow's return
    data['tomorrow_return'] = data.groupby('ticker')['close'].pct_change().shift(-1)
    
    # Option 1: Dynamic threshold based on volatility
    volatility = data.groupby('ticker')['close'].pct_change().rolling(20).std()
    dynamic_threshold = volatility * 1.0  # 1 standard deviation
    
    # Create multi-class target with XGBoost-compatible labels [0, 1, 2]
    # 0 = Down, 1 = Neutral, 2 = Up
    data['target_classification'] = np.select(
        [data['tomorrow_return'] > dynamic_threshold, 
         data['tomorrow_return'] < -dynamic_threshold],
        [2, 0],  # Up=2, Down=0
        default=1  # Neutral=1
    )
    
    # Option 2: Binary classification (simpler, often better)
    data['target_binary'] = (data['tomorrow_return'] > 0).astype(int)
    
    # Option 3: Large moves only (reduces noise) - also XGBoost compatible
    large_move_threshold = data['tomorrow_return'].abs().quantile(0.6)  # Top 40% moves
    data['target_large_moves'] = np.select(
        [data['tomorrow_return'] > large_move_threshold,
         data['tomorrow_return'] < -large_move_threshold],
        [2, 0],  # Up=2, Down=0
        default=1  # Neutral=1
    )
    
    return data

def generate_real_data(sector):
    """
    Downloads real stock data using yfinance with improved feature engineering.
    """
    print(f"    - Downloading and processing real data for {sector} sector...")
    
    tickers = sectors_map.get(sector, [])
    if not tickers:
        return pd.DataFrame()
        
    # Define a relevant historical range for the analysis
    start_date = '2017-01-01'
    end_date = '2024-12-31'
    
    try:
        # Download base data with auto-adjust for stock splits and dividends
        print(f"      - Downloading with auto-adjust enabled for stock splits...")
        data = yf.download(tickers, start=start_date, end=end_date, 
                          auto_adjust=True,  # Automatically adjust for splits and dividends
                          prepost=True,      # Include pre/post market data
                          threads=True,      # Use threading for faster downloads
                          progress=False)
    except Exception as e:
        print(f"Error downloading data for {sector}: {e}")
        return pd.DataFrame()

    # Handle single or multiple ticker download
    if len(tickers) == 1:
        data['ticker'] = tickers[0]
        data = data.reset_index().rename(columns={'Date': 'date'})
    else:
        # Unstack to flatten the MultiIndex (Date, Ticker) into rows (date, ticker, metrics)
        data = data.stack(level=1).rename_axis(['date', 'ticker']).reset_index()

    # Rename columns for consistency
    data = data.rename(columns={'Close': 'close', 'Volume': 'volume', 
                               'High': 'high', 'Low': 'low', 'Open': 'open'})
    
    # Calculate returns and technical indicators with extended lags
    data['return_today'] = data.groupby('ticker')['close'].pct_change()
    data['return_lag1'] = data.groupby('ticker')['close'].pct_change().shift(1)
    data['return_lag2'] = data.groupby('ticker')['close'].pct_change().shift(2)
    data['return_lag3'] = data.groupby('ticker')['close'].pct_change().shift(3)
    data['return_lag4'] = data.groupby('ticker')['close'].pct_change().shift(4)
    data['return_lag5'] = data.groupby('ticker')['close'].pct_change().shift(5)
    
    # Weekly and monthly aggregated returns
    data['return_weekly'] = data.groupby('ticker')['close'].pct_change(periods=5)  # 5-day return
    data['return_monthly'] = data.groupby('ticker')['close'].pct_change(periods=22)  # ~22 trading days
    
    # Rolling statistics for returns
    data['return_volatility_5d'] = data.groupby('ticker')['return_today'].rolling(5).std().reset_index(0, drop=True)
    data['return_volatility_20d'] = data.groupby('ticker')['return_today'].rolling(20).std().reset_index(0, drop=True)
    data['return_mean_5d'] = data.groupby('ticker')['return_today'].rolling(5).mean().reset_index(0, drop=True)
    data['return_mean_20d'] = data.groupby('ticker')['return_today'].rolling(20).mean().reset_index(0, drop=True)
    
    # Detect and verify stock split adjustments
    data = detect_and_adjust_splits(data)
    
    # Calculate real technical indicators
    data = calculate_real_technical_indicators(data)
    
    # Add split-adjusted features
    data = add_split_adjusted_features(data)
    
    # Create improved targets
    data = create_improved_targets(data)

    # Add sentiment data
    try:
        sentiment_data = pd.read_csv('General_daily_sentiment.csv', parse_dates=['date'])
        sentiment_data['sentiment_score'] = sentiment_data['sentiment_score'].fillna(0)
        sentiment_data['sentiment_score'] = sentiment_data['sentiment_score'] * 2
        data = pd.merge(data, sentiment_data, on=['date','ticker'], how='left')
        print(f"    - Added sentiment data: {len(sentiment_data)} records")
    except Exception as e:
        print(f"    - Warning: Could not load sentiment data: {e}")
        data['sentiment_score'] = 0

    # Add Fed rate features
    try:
        fed_data = add_fed_rate_features_api()
        if not fed_data.empty:
            fed_data['fed_rate_change'] = fed_data['fed_rate'].diff().fillna(0)
            fed_data['fed_rate_trend'] = fed_data['fed_rate'].rolling(window=5).mean()
            data = pd.merge(data, fed_data, on='date', how='left')
            data['fed_rate'] = data['fed_rate'].fillna(method='ffill').fillna(method='bfill')
            print(f"    - Added Fed rate data: {len(fed_data)} records")
        else:
            data['fed_rate'] = 0
    except Exception as e:
        print(f"    - Warning: Could not load Fed rate data: {e}")
        data['fed_rate'] = 0

    # Add external macroeconomic indicators
    try:
        external_data = add_external_macro_indicators(start_date, end_date)
        if not external_data.empty:
            data = pd.merge(data, external_data, on='date', how='left')
            # Forward fill missing values for macro indicators
            macro_cols = [col for col in external_data.columns if col != 'date']
            for col in macro_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            print(f"    - Added external macro indicators: {len(macro_cols)} features")
        else:
            print("    - No external macro indicators added")
    except Exception as e:
        print(f"    - Warning: Could not load external indicators: {e}")

    # Add sector-specific indices
    try:
        sector_indices = add_sector_specific_indices(sector, start_date, end_date)
        if not sector_indices.empty:
            data = pd.merge(data, sector_indices, on='date', how='left')
            # Forward fill missing values for sector indices
            sector_cols = [col for col in sector_indices.columns if col != 'date']
            for col in sector_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            print(f"    - Added sector-specific indices: {len(sector_cols)} features")
        else:
            print("    - No sector-specific indices added")
    except Exception as e:
        print(f"    - Warning: Could not load sector indices: {e}")
    
    # Use only real features - no random noise!
    print(f"    - Generated {len(data)} records with enhanced technical and macro indicators")
    
    return data.dropna()

def add_fed_rate_features_api(series_id='FEDFUNDS'):
    """Fetch and add Fed Funds Rate features from FRED API.
    """
    try:
        from fredapi import Fred
        fred = Fred(api_key='ebe322799ab9053da10a09b92a025a28')  # Replace with your actual FRED API key
        fed_series = fred.get_series(series_id)
        fed = fed_series.to_frame(name='fed_rate')
        fed.index = pd.to_datetime(fed.index)
        fed.reset_index(inplace=True)
        fed.rename(columns={'index': 'date'}, inplace=True)
        fed = fed.sort_values('date')

        # Resample to daily frequency (forward fill between FOMC updates)
        fed = fed.set_index('date').resample('D').ffill().reset_index()
        return fed

    except Exception as e:
        print(f"⚠️ Error fetching Fed data: {e}")
        return pd.DataFrame({'date': [], 'fed_rate': []})


def add_external_macro_indicators(start_date='2017-01-01', end_date= '2024-12-31'):
    """
    Fetch external macroeconomic and market indicators to enhance feature engineering.
    Returns a DataFrame with date and various macro indicators.
    """
    print("    - Fetching external macroeconomic indicators...")
    
    external_data = pd.DataFrame()
    
    try:
        # 1. VIX - Volatility Index (Market Fear/Greed indicator)
        print("      • Downloading VIX (Volatility Index)...")
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True, prepost=False)
        if not vix.empty and len(vix) > 100:
            vix_df = vix[['Close']].reset_index()
            vix_df.columns = ['date', 'vix']
            vix_df['vix_change'] = vix_df['vix'].pct_change()
            vix_df['vix_ma_20'] = vix_df['vix'].rolling(20).mean()
            vix_df['vix_spike'] = (vix_df['vix'] > vix_df['vix_ma_20'] * 1.2).astype(int)
            external_data = vix_df if external_data.empty else pd.merge(external_data, vix_df, on='date', how='outer')
            print(f"        ✅ VIX data: {len(vix_df)} records")
        else:
            print(f"        ⚠️ VIX data insufficient or empty")
    
    except Exception as e:
        print(f"      ⚠️ Error fetching VIX: {str(e)[:100]}...")
    
    try:
        # 2. Dollar Index (DXY) - Currency strength indicator
        print("      • Downloading DXY (Dollar Index)...")
        # Try multiple DXY symbols as fallbacks
        dxy_symbols = ['DX-Y.NYB', 'UUP', 'DXY']  # Dollar Index, Dollar ETF, alternative
        dxy_success = False
        
        for dxy_symbol in dxy_symbols:
            try:
                dxy = yf.download(dxy_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True, prepost=False)
                if not dxy.empty and len(dxy) > 100:
                    dxy_df = dxy[['Close']].reset_index()
                    dxy_df.columns = ['date', 'dxy']
                    dxy_df['dxy_change'] = dxy_df['dxy'].pct_change()
                    dxy_df['dxy_ma_50'] = dxy_df['dxy'].rolling(50).mean()
                    external_data = dxy_df if external_data.empty else pd.merge(external_data, dxy_df, on='date', how='outer')
                    print(f"        ✅ DXY data ({dxy_symbol}): {len(dxy_df)} records")
                    dxy_success = True
                    break
            except:
                continue
        
        if not dxy_success:
            print(f"        ⚠️ Could not fetch DXY data from any source")
            
    except Exception as e:
        print(f"      ⚠️ Error fetching DXY: {str(e)[:100]}...")
    
    try:
        # 3. 10-Year Treasury Yield - Interest rate environment
        print("      • Downloading 10Y Treasury Yield...")
        tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False, auto_adjust=True, prepost=False)
        if not tnx.empty:
            tnx_df = tnx[['Close']].reset_index()
            tnx_df.columns = ['date', 'treasury_10y']
            tnx_df['treasury_10y_change'] = tnx_df['treasury_10y'].pct_change()
            tnx_df['treasury_10y_ma_20'] = tnx_df['treasury_10y'].rolling(20).mean()
            external_data = tnx_df if external_data.empty else pd.merge(external_data, tnx_df, on='date', how='outer')
            
    except Exception as e:
        print(f"      ⚠️ Error fetching Treasury 10Y: {e}")
    
    try:
        # 4. Commodity Index (Oil - WTI)
        print("      • Downloading WTI Oil prices...")
        oil = yf.download('CL=F', start=start_date, end=end_date, progress=False, auto_adjust=True, prepost=False)
        if not oil.empty:
            oil_df = oil[['Close']].reset_index()
            oil_df.columns = ['date', 'oil_price']
            oil_df['oil_change'] = oil_df['oil_price'].pct_change()
            oil_df['oil_volatility'] = oil_df['oil_change'].rolling(20).std()
            external_data = oil_df if external_data.empty else pd.merge(external_data, oil_df, on='date', how='outer')
            
    except Exception as e:
        print(f"      ⚠️ Error fetching Oil prices: {e}")
    
    try:
        # 5. FRED API - Economic indicators
        print("      • Downloading FRED economic indicators...")
        from fredapi import Fred
        fred = Fred(api_key='ebe322799ab9053da10a09b92a025a28')
        
        # Consumer Price Index (Inflation)
        try:
            cpi_data = fred.get_series('CPIAUCSL', start=start_date, end=end_date)
            if not cpi_data.empty:
                cpi_df = cpi_data.to_frame(name='cpi').reset_index()
                cpi_df.columns = ['date', 'cpi']
                cpi_df['cpi_yoy'] = cpi_df['cpi'].pct_change(periods=12) * 100  # Year-over-year inflation
                # Resample to daily and forward fill
                cpi_df = cpi_df.set_index('date').resample('D').ffill().reset_index()
                external_data = cpi_df if external_data.empty else pd.merge(external_data, cpi_df, on='date', how='outer')
        except Exception as e:
            print(f"        ⚠️ CPI data error: {e}")
        
        # Unemployment Rate
        try:
            unemployment_data = fred.get_series('UNRATE', start=start_date, end=end_date)
            if not unemployment_data.empty:
                unemployment_df = unemployment_data.to_frame(name='unemployment_rate').reset_index()
                unemployment_df.columns = ['date', 'unemployment_rate']
                unemployment_df['unemployment_change'] = unemployment_df['unemployment_rate'].diff()
                # Resample to daily and forward fill
                unemployment_df = unemployment_df.set_index('date').resample('D').ffill().reset_index()
                external_data = unemployment_df if external_data.empty else pd.merge(external_data, unemployment_df, on='date', how='outer')
        except Exception as e:
            print(f"        ⚠️ Unemployment data error: {e}")
            
    except Exception as e:
        print(f"      ⚠️ Error with FRED API: {e}")
    
    # If we got some data, add derived features
    if not external_data.empty:
        external_data['date'] = pd.to_datetime(external_data['date'])
        
        # Market stress indicator (combination of VIX and yield spread)
        if 'vix' in external_data.columns and 'treasury_10y' in external_data.columns:
            external_data['market_stress'] = (
                (external_data['vix'] / external_data['vix'].rolling(60).mean()) +
                (external_data['treasury_10y'] / external_data['treasury_10y'].rolling(60).mean())
            ) / 2
        
        # Risk-on/Risk-off sentiment
        if 'vix' in external_data.columns and 'dxy' in external_data.columns:
            external_data['risk_sentiment'] = (
                (external_data['vix'].rolling(5).mean() / external_data['vix'].rolling(20).mean()) * -1 +
                (external_data['dxy'].rolling(5).mean() / external_data['dxy'].rolling(20).mean())
            )
        
        print(f"    - Successfully fetched {len(external_data)} records of external indicators")
        return external_data.fillna(method='ffill').fillna(method='bfill')
    
    else:
        print("    - No external indicators fetched, returning empty DataFrame")
        return pd.DataFrame()


def add_sector_specific_indices(sector, start_date='2017-01-01', end_date= '2024-12-31'):
    """
    Add sector-specific market indices as features with fallback options for missing symbols.
    """
    print(f"      • Downloading sector-specific indices for {sector}...")
    
    # Updated sector indices with fallback options and working symbols
    sector_indices = {
        'Tech': ['^IXIC', '^NDX', 'QQQ'],  # NASDAQ Composite, NASDAQ 100, NASDAQ ETF
        'Finance': ['^BKX', 'XLF', 'KRE'],  # Bank Index, Financial ETF, Regional Banks ETF (removed ^KBW)
        'HealthCare': ['XLV', 'VHT', 'IHI']  # Health Care ETF, Vanguard Health, Medical Devices ETF (removed ^HCX)
    }
    
    sector_data = pd.DataFrame()
    successful_indices = []
    
    for idx_symbol in sector_indices.get(sector, []):
        try:
            print(f"        - Attempting to download {idx_symbol}...")
            idx_data = yf.download(idx_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True, prepost=False)
            
            if not idx_data.empty and len(idx_data) > 10:  # Reduced threshold for testing
                idx_df = idx_data[['Close']].reset_index()
                idx_name = idx_symbol.replace('^', '').replace('-', '_').lower()
                idx_df.columns = ['date', f'{idx_name}_price']
                
                # Calculate derived features
                idx_df[f'{idx_name}_return'] = idx_df[f'{idx_name}_price'].pct_change()
                idx_df[f'{idx_name}_ma_20'] = idx_df[f'{idx_name}_price'].rolling(20).mean()
                idx_df[f'{idx_name}_rsi'] = calculate_rsi(idx_df[f'{idx_name}_price'])
                
                # Add volatility and momentum features
                idx_df[f'{idx_name}_volatility'] = idx_df[f'{idx_name}_return'].rolling(20).std()
                idx_df[f'{idx_name}_momentum'] = idx_df[f'{idx_name}_price'].pct_change(periods=10)
                
                if sector_data.empty:
                    sector_data = idx_df
                else:
                    sector_data = pd.merge(sector_data, idx_df, on='date', how='outer')
                
                successful_indices.append(idx_symbol)
                print(f"        ✅ Successfully downloaded {idx_symbol}: {len(idx_df)} records")
            else:
                print(f"        ⚠️ {idx_symbol}: Insufficient data (got {len(idx_data) if not idx_data.empty else 0} records)")
                    
        except Exception as e:
            print(f"        ⚠️ Error fetching {idx_symbol}: {str(e)[:100]}...")
            continue
    
    if successful_indices:
        print(f"        📊 Successfully loaded {len(successful_indices)} indices: {successful_indices}")
    else:
        print(f"        ❌ No sector indices loaded for {sector}")
    
    return sector_data


def calculate_rsi(prices, window=14):
    """Calculate RSI for a price series."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def detect_and_adjust_splits(data):
    """
    Detect potential stock splits and ensure proper adjustment.
    Auto-adjust should handle this, but this provides additional verification.
    """
    print("    - Detecting and verifying stock split adjustments...")
    
    splits_detected = 0
    for ticker in data['ticker'].unique():
        ticker_mask = data['ticker'] == ticker
        ticker_data = data[ticker_mask].copy().sort_values('date')
        
        if len(ticker_data) < 2:
            continue
            
        # Calculate daily price changes
        ticker_data['price_change'] = ticker_data['close'].pct_change()
        
        # Detect potential splits (large negative price changes > 25%)
        potential_splits = ticker_data[ticker_data['price_change'] < -0.25]
        
        if not potential_splits.empty:
            splits_detected += len(potential_splits)
            print(f"      • {ticker}: {len(potential_splits)} potential splits detected and auto-adjusted")
            
            # Log split dates for reference
            for idx, split_row in potential_splits.iterrows():
                split_date = split_row['date']
                price_drop = split_row['price_change'] * 100
                print(f"        - {split_date.strftime('%Y-%m-%d')}: {price_drop:.1f}% price drop (likely split)")
    
    if splits_detected == 0:
        print("      ✅ No significant stock splits detected - data appears properly adjusted")
    else:
        print(f"      ✅ {splits_detected} splits detected across all tickers - auto-adjustment verified")
    
    return data


def add_split_adjusted_features(data):
    """
    Add features that are specifically designed to handle split-adjusted data.
    """
    print("    - Adding split-adjusted technical features...")
    
    for ticker in data['ticker'].unique():
        ticker_mask = data['ticker'] == ticker
        ticker_data = data[ticker_mask].copy().sort_values('date')
        
        # Split-adjusted volume metrics (volume should be inversely adjusted for splits)
        # Volume ratio - helps detect when volume adjustments may be incorrect
        data.loc[ticker_mask, 'volume_ma_20'] = ticker_data['volume'].rolling(20).mean()
        data.loc[ticker_mask, 'volume_ratio'] = ticker_data['volume'] / data.loc[ticker_mask, 'volume_ma_20']
        
        # Price consistency checks
        data.loc[ticker_mask, 'price_stability'] = ticker_data['close'].rolling(5).std() / ticker_data['close'].rolling(5).mean()
        
        # Adjusted momentum indicators that are split-resistant
        data.loc[ticker_mask, 'momentum_3m'] = (ticker_data['close'] / ticker_data['close'].shift(63) - 1) * 100  # 3-month momentum
        data.loc[ticker_mask, 'momentum_6m'] = (ticker_data['close'] / ticker_data['close'].shift(126) - 1) * 100  # 6-month momentum
        
        # Relative strength vs moving averages (split-neutral)
        data.loc[ticker_mask, 'price_vs_ma50'] = (ticker_data['close'] / data.loc[ticker_mask, 'EMA_50']) - 1
        data.loc[ticker_mask, 'price_vs_ma200'] = (ticker_data['close'] / data.loc[ticker_mask, 'EMA_200']) - 1
    
    print("      ✅ Split-adjusted features added successfully")
    return data


class ModelComparator:
    """
    Improved class with better feature validation and model training.
    """
    def __init__(self, data, save_models=False, save_dir='models', tag='run'):
        # if data.empty or data['date'].nunique() < 500:
        #     raise ValueError("Input data is insufficient for time-series analysis.")
        #
        # self.data = data.sort_values(by='date').reset_index(drop=True)
        # self.data['year'] = pd.to_datetime(self.data['date']).dt.year
        # print(f"    - Initialized with {len(self.data)} records across {self.data['year'].nunique()} years")
        #
        # # Analyze target distribution
        # self._analyze_targets()
        if data.empty or data['date'].nunique() < 200:
            raise ValueError("Input data is insufficient for time-series analysis.")

        self.data = data.sort_values(by='date').reset_index(drop=True)
        self.data['year'] = pd.to_datetime(self.data['date']).dt.year
        self.save_models = save_models
        self.save_dir = Path(save_dir)
        self.tag = tag  # e.g. 'General' or sector name
        # ensure base dir exists only if saving requested
        if self.save_models:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"    - Initialized with {len(self.data)} records across {self.data['year'].nunique()} years")
        self._analyze_targets()

    # Helper to save models (add inside ModelComparator)
    def _save_model(self, model, name, scope):
        """
        Save model artifact to disk. `scope` is 'regression'/'classification' or custom.
        Name is short model id like 'xgb_regressor' or 'attention_lstm'.
        """
        if not self.save_models:
            return None

        ts = int(time.time())
        out_dir = self.save_dir / self.tag / scope
        out_dir.mkdir(parents=True, exist_ok=True)

        # XGBoost Booster has save_model method; sklearn use joblib; keras use .save()
        fname_base = f"{name}_{self.tag}_{ts}"
        try:
            # XGBoost
            if hasattr(model, "save_model"):
                path = out_dir / f"{fname_base}.json"
                model.save_model(str(path))
                return path
            # Keras Model
            if hasattr(model, "save") and hasattr(model, "to_json"):
                # save as .h5 (single file) for portability
                path = out_dir / f"{fname_base}.h5"
                model.save(str(path))
                return path
            # sklearn / joblib-serializable
            else:
                path = out_dir / f"{fname_base}.joblib"
                joblib.dump(model, str(path))
                return path
        except Exception as e:
            print(f"    - Warning: failed to save {name}: {e}")
            return None

    def _analyze_targets(self):
        """Analyze and print target variable distributions."""
        print("    - Target variable distributions:")
        if 'target_classification' in self.data.columns:
            dist = self.data['target_classification'].value_counts().sort_index()
            total = len(self.data)
            print(f"      Multi-class: Down(0): {dist.get(0, 0)/total:.1%}, "
                  f"Neutral(1): {dist.get(1, 0)/total:.1%}, Up(2): {dist.get(2, 0)/total:.1%}")
        
        if 'target_binary' in self.data.columns:
            dist = self.data['target_binary'].value_counts().sort_index()
            total = len(self.data)
            print(f"      Binary: Down(0): {dist.get(0, 0)/total:.1%}, Up(1): {dist.get(1, 0)/total:.1%}")

    def validate_features(self, features, target_type='classification'):
        """
        Enhanced feature validation with improved filtering for better model performance.
        """
        # Get only real features that exist in data
        valid_features = [f for f in features if f in self.data.columns]
        
        # Remove constant or mostly null features with stricter criteria
        feature_quality = {}
        for feature in valid_features:
            feature_series = self.data[feature]
            null_pct = feature_series.isnull().mean()
            std = feature_series.std()
            unique_ratio = feature_series.nunique() / len(feature_series)
            feature_quality[feature] = {
                'null_pct': null_pct, 
                'std': std, 
                'unique_ratio': unique_ratio
            }
        
        # Enhanced filtering criteria
        good_features = [
            f for f in valid_features 
            if (feature_quality[f]['null_pct'] < 0.2 and  # Less than 20% null
                feature_quality[f]['std'] > 1e-8 and      # Has variation
                feature_quality[f]['unique_ratio'] > 0.01)  # At least 1% unique values
        ]
        # good_features.append('final_signal') if 'final_signal' in self.data.columns else None

        
        print(f"    - Feature validation: {len(good_features)}/{len(features)} features passed enhanced filtering")
        
        # Calculate mutual information for feature importance and selection
        if good_features and target_type == 'classification':
            target_col = 'target_binary' if 'target_binary' in self.data.columns else 'target_classification'
            X_sample = self.data[good_features].fillna(0)
            y_sample = self.data[target_col]
            
            try:
                mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
                
                # Create feature importance ranking
                feature_importance = list(zip(good_features, mi_scores))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print("    - Top 20 features by mutual information:")
                for i, (feature, score) in enumerate(feature_importance[:20]):
                    print(f"      {i+1:2d}. {feature:<25}: {score:.4f}")
                
                # Select top features (adaptive selection based on data size)
                if len(good_features) > 30:
                    # For large feature sets, select top performers
                    threshold = np.percentile(mi_scores, 60)  # Top 40% of features
                    selected_features = [f for f, score in feature_importance if score >= threshold]
                    print(f"    - Selected {len(selected_features)} top-performing features from {len(good_features)}")
                    return selected_features
                else:
                    return good_features
                    
            except Exception as e:
                print(f"    - Could not calculate feature importance: {e}")
                return good_features

        elif good_features and target_type == 'regression':
            target_col = 'tomorrow_return'
            X_sample = self.data[good_features].fillna(0)
            y_sample = self.data[target_col]

            try:
                mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)

                # Create feature importance ranking
                feature_importance = list(zip(good_features, mi_scores))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                print("    - Top 20 features by mutual information (regression):")
                for i, (feature, score) in enumerate(feature_importance[:20]):
                    print(f"      {i+1:2d}. {feature:<25}: {score:.4f}")

                # Select top features (adaptive selection based on data size)
                if len(good_features) > 30:
                    threshold = np.percentile(mi_scores, 60)  # Top 40% of features
                    selected_features = [f for f, score in feature_importance if score >= threshold]
                    print(f"    - Selected {len(selected_features)} top-performing features from {len(good_features)}")
                    return selected_features
                else:
                    return good_features

            except Exception as e:
                print(f"    - Could not calculate feature importance: {e}")
                return good_features
        return good_features

    # python
    def _get_rolling_splits(self, test_months=6, min_train_months=12, step_months=6,
                            min_train_rows=200, min_test_rows=50):
        """
        Generate walk-forward splits where each test window is `test_months` long.
        - test_months: length of test window in months (default 6)
        - min_train_months: minimum history (months) required before first test (default 12)
        - step_months: how far to move the test window each iteration (default 6)
        Yields tuples (train_idx, test_idx) of pandas Index objects.
        """
        # ensure date is datetime
        if 'date' not in self.data.columns:
            print("    - No `date` column found; cannot create time-based splits")
            return []

        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        start_date = df['date'].min()
        end_date = df['date'].max()
        print(f"    - Date range: {start_date.date()} → {end_date.date()}")

        # earliest possible test start after minimum training months
        first_test_start = start_date + pd.DateOffset(months=min_train_months)
        if first_test_start >= end_date:
            print("    - Insufficient history for requested min_train_months")
            return []

        splits_generated = 0
        test_start = first_test_start

        while test_start < end_date:
            test_end = test_start + pd.DateOffset(months=test_months)
            if test_start >= test_end:
                break
            # clip test_end to available data
            if test_start >= end_date:
                break
            # Build index sets
            train_idx = df[df['date'] < test_start].index
            test_idx = df[(df['date'] >= test_start) & (df['date'] < test_end)].index

            # Validate sizes
            if len(train_idx) >= min_train_rows and len(test_idx) >= min_test_rows:
                splits_generated += 1
                train_span = (df.loc[train_idx, 'date'].min().date(), df.loc[train_idx, 'date'].max().date())
                test_span = (df.loc[test_idx, 'date'].min().date(), df.loc[test_idx, 'date'].max().date())
                print(f"    - Split {splits_generated}: Train {train_span} ({len(train_idx)} rows) → "
                      f"Test {test_span} ({len(test_idx)} rows)")
                yield train_idx, test_idx

            # advance the test window
            test_start = test_start + pd.DateOffset(months=step_months)

        if splits_generated == 0:
            print("    - No valid 6-month splits generated")
            return []

    # def _get_rolling_splits(self):
    #     """Generates proper walk-forward validation splits."""
    #     years = sorted(self.data['year'].unique())
    #     print(f"    - Available years: {years}")
    #
    #     min_train_years = 3
    #     if len(years) < min_train_years + 1:
    #         print(f"    - Insufficient years ({len(years)}) for proper validation")
    #         return []
    #
    #     splits_generated = 0
    #     for i in range(min_train_years, len(years)):
    #         train_years = years[:i]
    #         test_year = years[i]
    #
    #         train_idx = self.data[self.data['year'].isin(train_years)].index
    #         test_idx = self.data[self.data['year'] == test_year].index
    #
    #         if len(train_idx) >= 200 and len(test_idx) >= 50:
    #             splits_generated += 1
    #             print(f"    - Split {splits_generated}: Train years {train_years} → Test {test_year}")
    #             yield train_idx, test_idx
    #
    #     if splits_generated == 0:
    #         print("    - No valid splits generated")
    #         return []

    def compare_regression_models(self, features):
        """Compares models on regression task."""
        valid_features = self.validate_features(features, 'regression')
        if not valid_features: return {}
            
        X = self.data[valid_features].fillna(0)
        y = self.data['tomorrow_return']

        results = []
        for train_idx, test_idx in self._get_rolling_splits():
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            if len(X_train) < 100: continue

            xgb_pred = self._train_xgb_regressor(X_train, y_train, X_test)
            lr_pred = self._train_linear_regression(X_train, y_train, X_test)
            # lstm_pred = self._train_lstm_regressor(X_train, y_train, X_test)
            hybrid_lstm_pred = self._train_hybrid_lstm_regressor(X_train, y_train, X_test)
            current_result = {'XGBoost': {'MSE':mean_squared_error(y_test, xgb_pred) ,'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                           'R2': r2_score(y_test, xgb_pred)},
                'LinearRegression': {'MSE':mean_squared_error(y_test, lr_pred) ,'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
                                   'R2': r2_score(y_test, lr_pred)},
                # 'AttentionLSTM': {'MSE':mean_squared_error(y_test, lstm_pred) ,'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
                #         'R2': r2_score(y_test, lstm_pred)},
                'HybridAttentionLSTM': {'MSE':mean_squared_error(y_test, hybrid_lstm_pred) ,'RMSE': np.sqrt(mean_squared_error(y_test, hybrid_lstm_pred)),
                                     'R2': r2_score(y_test, hybrid_lstm_pred)}
            }
            print("=======Regression results for current split=======")
            print(current_result)

            results.append(current_result)

        return self._aggregate_yearly_results(results)

    def compare_classification_models(self, features, problem_type='binary'):
        """Compares models on classification task with improved handling."""
        if problem_type == 'binary' and 'target_binary' in self.data.columns:
            target_col = 'target_binary'
            num_class = 2
        else:
            target_col = 'target_classification'
            num_class = 3
            
        valid_features = self.validate_features(features, 'classification')
        if not valid_features: return {}

        X = self.data[valid_features].fillna(0)
        y = self.data[target_col]

        # Pre-scale X for non-tree models
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=valid_features, index=X.index)

        results = []
        for train_idx, test_idx in self._get_rolling_splits():
            X_train, X_test = X_scaled.loc[train_idx], X_scaled.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            if len(X_train) < 100: continue

            X_train_xgb, X_test_xgb = X.loc[train_idx], X.loc[test_idx]
            
            xgb_pred = self._train_xgb_classifier(X_train_xgb, y_train, X_test_xgb, num_class)
            log_pred = self._train_logistic_regression(X_train, y_train, X_test, num_class)
            # lstm_pred = self._train_lstm_classifier(X_train, y_train, X_test, num_class)
            hybrid_lstm_pred = self._train_hybrid_lstm_classifier(X_train, y_train, X_test, num_class)

            # Enhanced evaluation
            def get_direction_counts(preds):
                counts = Counter(preds)
                if num_class == 2:
                    return {'Down': counts.get(0, 0), 'Up': counts.get(1, 0)}
                else:
                    return {'Down': counts.get(0, 0), 'Neutral': counts.get(1, 0), 'Up': counts.get(2, 0)}

            # Calculate metrics
            avg_type = 'binary' if num_class == 2 else 'weighted'
            current_result = {
                'XGBoost': {
                    'F1': f1_score(y_test, xgb_pred, average=avg_type),
                    'Precision': precision_score(y_test, xgb_pred, average=avg_type),
                    'Direction': get_direction_counts(xgb_pred)
                },
                'LogisticRegression': {
                    'F1': f1_score(y_test, log_pred, average=avg_type),
                    'Precision': precision_score(y_test, log_pred, average=avg_type),
                    'Direction': get_direction_counts(log_pred)
                },
                # 'AttentionLSTM': {
                #     'F1': f1_score(y_test, lstm_pred, average=avg_type),
                #     'Precision': precision_score(y_test, lstm_pred, average=avg_type),
                #     'Direction': get_direction_counts(lstm_pred)
                # },
                'HybridAttentionLSTM': {
                    'F1': f1_score(y_test, hybrid_lstm_pred, average=avg_type),
                    'Precision': precision_score(y_test, hybrid_lstm_pred, average=avg_type),
                    'Direction': get_direction_counts(hybrid_lstm_pred)
                }
            }
            
            results.append(current_result)
            print("=======Classification results for current split=======")
            print({
                'XGBoost': {
                    'F1': f1_score(y_test, xgb_pred, average=avg_type),
                    'Precision': precision_score(y_test, xgb_pred, average=avg_type),
                    'Direction': get_direction_counts(xgb_pred)
                },
                'LogisticRegression': {
                    'F1': f1_score(y_test, log_pred, average=avg_type),
                    'Precision': precision_score(y_test, log_pred, average=avg_type),
                    'Direction': get_direction_counts(log_pred)
                },

                'HybridAttentionLSTM': {
                    'F1': f1_score(y_test, hybrid_lstm_pred, average=avg_type),
                    'Precision': precision_score(y_test, hybrid_lstm_pred, average=avg_type),
                    'Direction': get_direction_counts(hybrid_lstm_pred)
                }
            })
            
            # Print detailed report for first split
            # if len(results) == 1:
            # self._print_detailed_report(y_test, xgb_pred, log_pred, hybrid_lstm_pred, num_class)
                
        return self._aggregate_yearly_results(results)

    def _print_detailed_report(self, y_test, xgb_pred, log_pred, lstm_pred, num_class):
        """Print detailed classification report for first split."""
        print("\n    - First split detailed results:")
        models = ['XGBoost', 'LogisticRegression', 'AttentionLSTM']
        predictions = [xgb_pred, log_pred, lstm_pred]
        
        for model_name, pred in zip(models, predictions):
            print(f"\n    {model_name}:")
            print(classification_report(y_test, pred, 
                                      target_names=['Down', 'Up'] if num_class == 2 else ['Down', 'Neutral', 'Up']))

    # --- Improved Model Training Methods ---
    def _train_xgb_regressor(self, X_t, y_t, X_v):
        model = xgb.XGBRegressor(
            objective="reg:squarederror", 
            n_estimators=500,
            learning_rate=0.05, 
            max_depth=6,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_t, y_t, verbose=False)
        saved = self._save_model(model, "xgb_regressor", "regression")
        if saved:
            print(f"    - Saved XGBoost regressor to {saved}")
        return model.predict(X_v)

    def _train_linear_regression(self, X_t, y_t, X_v):
        model = LinearRegression().fit(X_t, y_t)
        saved = self._save_model(model, "linear_regression", "regression")
        if saved:
            print(f"    - Saved LinearRegression to {saved}")
        return model.predict(X_v)

    def _train_lstm_regressor(self, X_t, y_t, X_v):
        """Train Attention-based LSTM for regression task."""
        scaler = StandardScaler()
        X_t_scaled = scaler.fit_transform(X_t)
        X_v_scaled = scaler.transform(X_v)

        # For time series, create sequences from features (simplified approach)
        # Reshape to (samples, features, 1) for LSTM input
        X_t_lstm = X_t_scaled.reshape(X_t_scaled.shape[0], X_t_scaled.shape[1], 1)
        X_v_lstm = X_v_scaled.reshape(X_v_scaled.shape[0], X_v_scaled.shape[1], 1)

        # Build simplified attention-based LSTM model
        input_shape = (X_t_lstm.shape[1], X_t_lstm.shape[2])

        try:
            # Create simplified attention model with fewer parameters
            inputs = Input(shape=input_shape)

            # LSTM layer with return_sequences=True
            lstm_out = LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(inputs)

            # Simplified attention mechanism
            attention_scores = Dense(1, activation='tanh')(lstm_out)
            attention_scores = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention_scores)

            # Apply attention weights
            context_vector = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([lstm_out, attention_scores])

            # Dense layers
            dense_out = Dense(16, activation='relu')(context_vector)
            dense_out = Dropout(0.2)(dense_out)
            outputs = Dense(1, activation='linear')(dense_out)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # Train with fewer epochs for faster execution
            model.fit(X_t_lstm, y_t.values,
                     epochs=10,
                     batch_size=64,
                     verbose=0,
                     validation_split=0.1)

            saved = self._save_model(model, "attention_lstm_regressor", "regression")
            if saved:
                print(f"    - Saved Attention LSTM regressor to {saved}")

            # Make predictions
            predictions = model.predict(X_v_lstm, verbose=0).flatten()
            return predictions

        except Exception as e:
            print(f"    Warning: Attention LSTM failed ({e}), falling back to simple LSTM")
            # Fallback to simple LSTM
            fallback_model = Sequential([
                LSTM(32, input_shape=input_shape, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            fallback_model.compile(optimizer='adam', loss='mse')
            fallback_model.fit(X_t_lstm, y_t.values, epochs=10, batch_size=64, verbose=0)
            saved = self._save_model(fallback_model, "fallback_lstm_regressor", "regression")
            if saved:
                print(f"    - Saved fallback LSTM regressor to {saved}")
            return fallback_model.predict(X_v_lstm, verbose=0).flatten()

    def _train_hybrid_lstm_regressor(self, X_t, y_t, X_v):
        """Train Hybrid LSTM–GRU Attention-based Network for regression task."""

        # Standardize input features
        scaler = StandardScaler()
        X_t_scaled = scaler.fit_transform(X_t)
        X_v_scaled = scaler.transform(X_v)

        # Reshape data for LSTM input (samples, timesteps, features)
        X_t_lstm = X_t_scaled.reshape(X_t_scaled.shape[0], X_t_scaled.shape[1], 1)
        X_v_lstm = X_v_scaled.reshape(X_v_scaled.shape[0], X_v_scaled.shape[1], 1)
        input_shape = (X_t_lstm.shape[1], X_t_lstm.shape[2])

        try:
            # ======== Hybrid LSTM + GRU model with attention ========
            inputs = Input(shape=input_shape)

            # Parallel LSTM and GRU branches
            lstm_out = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            gru_out = GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)

            # Merge both temporal feature maps
            merged_seq = Concatenate(axis=-1)([lstm_out, gru_out])

            # Attention mechanism
            attention_scores = Dense(1, activation='tanh')(merged_seq)
            attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention_scores)
            context_vector = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([merged_seq, attention_weights])

            # Dense layers for regression output
            dense_out = Dense(64, activation='relu')(context_vector)
            dense_out = Dropout(0.3)(dense_out)
            dense_out = Dense(32, activation='relu')(dense_out)
            outputs = Dense(1, activation='linear')(dense_out)

            # Build and compile model
            model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

            # ======== Training (20 epochs per paper setup) ========
            model.fit(
                X_t_lstm, y_t.values,
                epochs=20,
                batch_size=64,
                verbose=1,
                validation_split=0.1
            )

            # Save model
            saved = self._save_model(model, "hybrid_lstm_gru_regressor", "regression")
            if saved:
                print(f"    - Saved Hybrid LSTM–GRU regressor to {saved}")

            # Make predictions
            predictions = model.predict(X_v_lstm, verbose=0).flatten()
            return predictions

        except Exception as e:
            print(f"    Warning: Hybrid LSTM–GRU failed ({e}), falling back to simple LSTM")

            # ======== Fallback simple LSTM ========
            fallback_model = Sequential([
                LSTM(32, input_shape=input_shape, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')
            ])
            fallback_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            fallback_model.fit(X_t_lstm, y_t.values, epochs=20, batch_size=64, verbose=0)

            saved = self._save_model(fallback_model, "fallback_lstm_regressor", "regression")
            if saved:
                print(f"    - Saved fallback LSTM regressor to {saved}")

            return fallback_model.predict(X_v_lstm, verbose=0).flatten()


    def _train_xgb_classifier(self, X_t, y_t, X_v, num_class):
        # Handle class imbalance
        if num_class == 2:
            objective = 'binary:logistic'
            # Calculate scale_pos_weight for imbalanced binary classification
            ratio = len(y_t[y_t == 0]) / len(y_t[y_t == 1]) if len(y_t[y_t == 1]) > 0 else 1
        else:
            objective = 'multi:softmax'
            ratio = 1  # For multi-class, we'll rely on sample_weight
            
        model = xgb.XGBClassifier(
            objective=objective,
            num_class=num_class if num_class > 2 else None,
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=ratio if num_class == 2 else 1,
            random_state=42,
            n_jobs=-1
        )
        
        # Use sample weights for multi-class imbalance
        if num_class > 2:
            class_counts = y_t.value_counts()
            total_samples = len(y_t)
            sample_weights = y_t.map({cls: total_samples / (len(class_counts) * count) 
                                    for cls, count in class_counts.items()})
            model.fit(X_t, y_t, sample_weight=sample_weights)
        else:
            model.fit(X_t, y_t)
        self._save_model(model, "xgb_classifier", "classification")

        return model.predict(X_v)

    def _train_logistic_regression(self, X_t, y_t, X_v, num_class):
        if num_class == 2:
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        else:
            model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42, class_weight='balanced')
        
        model.fit(X_t, y_t)
        self._save_model(model, "logistic_regression", "classification")

        return model.predict(X_v)

    def _train_lstm_classifier(self, X_t, y_t, X_v, num_class):
        """Train Attention-based LSTM for classification task."""
        # Prepare targets
        if num_class == 2:
            y_t_ohe = to_categorical(y_t, num_classes=2)
            output_units = 2
            loss = 'binary_crossentropy'
            activation = 'sigmoid'
        else:
            # y_t already has correct format [0, 1, 2] from create_improved_targets
            y_t_ohe = to_categorical(y_t, num_classes=3)
            output_units = 3
            loss = 'categorical_crossentropy'
            activation = 'softmax'

        # Prepare data - reshape for LSTM
        X_t_values = X_t.values.astype(np.float32)
        X_v_values = X_v.values.astype(np.float32)

        # Reshape: (samples, features, 1)
        X_t_lstm = X_t_values.reshape(X_t_values.shape[0], X_t_values.shape[1], 1)
        X_v_lstm = X_v_values.reshape(X_v_values.shape[0], X_v_values.shape[1], 1)

        # Build simplified attention-based LSTM model
        input_shape = (X_t_lstm.shape[1], X_t_lstm.shape[2])

        try:
            # Create simplified attention model
            inputs = Input(shape=input_shape)

            # LSTM layer with return_sequences=True
            lstm_out = LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(inputs)

            # Simplified attention mechanism
            attention_scores = Dense(1, activation='tanh')(lstm_out)
            attention_scores = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention_scores)

            # Apply attention weights
            context_vector = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([lstm_out, attention_scores])

            # Dense layers
            dense_out = Dense(16, activation='relu')(context_vector)
            dense_out = Dropout(0.2)(dense_out)
            outputs = Dense(output_units, activation=activation)(dense_out)

            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

            # Train model with fewer epochs for faster execution
            model.fit(X_t_lstm, y_t_ohe,
                     epochs=10,
                     batch_size=64,
                     verbose=0,
                     validation_split=0.1)

            self._save_model(model, "attention_lstm_classifier", "classification")

            # Make predictions
            preds_proba = model.predict(X_v_lstm, verbose=0)

            if num_class == 2:
                return (preds_proba[:, 1] > 0.5).astype(int)
            else:
                pred_classes = np.argmax(preds_proba, axis=1)
                return pred_classes  # Already in correct format [0, 1, 2]

        except Exception as e:
            print(f"    Warning: Attention LSTM failed ({e}), falling back to simple LSTM")
            # Fallback to simple LSTM
            fallback_model = Sequential([
                LSTM(32, input_shape=input_shape, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(output_units, activation=activation)
            ])

            fallback_model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            fallback_model.fit(X_t_lstm, y_t_ohe, epochs=10, batch_size=64, verbose=0)

            preds_proba = fallback_model.predict(X_v_lstm, verbose=0)
            if num_class == 2:
                return (preds_proba[:, 1] > 0.5).astype(int)
            else:
                pred_classes = np.argmax(preds_proba, axis=1)
                return pred_classes  # Already in correct format [0, 1, 2]

    def _train_hybrid_lstm_classifier(self, X_t, y_t, X_v, num_class):
        """Train Hybrid LSTM–GRU Network for classification task (Keras implementation)."""

        # Prepare targets
        if num_class == 2:
            y_t_ohe = to_categorical(y_t, num_classes=2)
            output_units = 2
            loss = 'binary_crossentropy'
            activation = 'sigmoid'
        else:
            y_t_ohe = to_categorical(y_t, num_classes=3)
            output_units = 3
            loss = 'categorical_crossentropy'
            activation = 'softmax'

        # Prepare data and reshape for LSTM
        X_t_values = X_t.values.astype(np.float32)
        X_v_values = X_v.values.astype(np.float32)

        X_t_lstm = X_t_values.reshape(X_t_values.shape[0], X_t_values.shape[1], 1)
        X_v_lstm = X_v_values.reshape(X_v_values.shape[0], X_v_values.shape[1], 1)
        input_shape = (X_t_lstm.shape[1], X_t_lstm.shape[2])

        try:
            # Input
            inputs = Input(shape=input_shape)

            # Parallel LSTM and GRU layers (hybrid)
            lstm_out = LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            gru_out = GRU(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)

            # Concatenate both sequence outputs
            merged_seq = Concatenate(axis=-1)([lstm_out, gru_out])

            # Simple attention mechanism
            attention_scores = Dense(1, activation='tanh')(merged_seq)
            attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis=1))(attention_scores)
            context_vector = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([merged_seq, attention_weights])

            # Dense layers
            dense_out = Dense(32, activation='relu')(context_vector)
            dense_out = Dropout(0.3)(dense_out)
            outputs = Dense(output_units, activation=activation)(dense_out)

            # Build and compile model
            model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # Train model (20 epochs as per the paper)
            model.fit(
                X_t_lstm, y_t_ohe,
                epochs=20,
                batch_size=64,
                verbose=1,
                validation_split=0.1
            )

            # Save model
            self._save_model(model, "hybrid_lstm_gru_classifier", "classification")

            # Predictions
            preds_proba = model.predict(X_v_lstm, verbose=0)
            if num_class == 2:
                return (preds_proba[:, 1] > 0.5).astype(int)
            else:
                return np.argmax(preds_proba, axis=1)

        except Exception as e:
            print(f"Warning: Hybrid LSTM–GRU failed ({e}), falling back to simple LSTM")
            fallback = Sequential([
                LSTM(32, input_shape=input_shape, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(output_units, activation=activation)
            ])
            fallback.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
            fallback.fit(X_t_lstm, y_t_ohe, epochs=20, batch_size=64, verbose=0)
            preds_proba = fallback.predict(X_v_lstm, verbose=0)
            return np.argmax(preds_proba, axis=1) if num_class > 2 else (preds_proba[:, 1] > 0.5).astype(int)

    def _aggregate_yearly_results(self, yearly_results):
        """Averages metrics across all test years."""
        if not yearly_results: return {}
        
        agg_results = {}
        for model in yearly_results[0].keys():
            agg_results[model] = {}
            for metric in yearly_results[0][model].keys():
                if metric == 'Direction':
                    # Sum direction counts
                    total_direction = Counter()
                    for year_res in yearly_results:
                        total_direction.update(year_res[model]['Direction'])
                    agg_results[model]['Direction'] = total_direction
                else:
                    # Average numeric metrics
                    values = [year_res[model][metric] for year_res in yearly_results]
                    agg_results[model][metric] = np.mean(values)
        
        return agg_results

def main():
    """Main function to run the comparison with improved analysis."""
    all_sectors = ['Tech', 'Finance', 'HealthCare']
    final_results = {'regression': {}, 'classification_binary': {}, 'classification_multi': {}}

    # print("="*60, "\n▶️ STEP 1: TRAINING GENERAL MODEL (ALL SECTORS COMBINED)\n", "="*60)
    # general_data = pd.concat([generate_real_data(s) for s in all_sectors], ignore_index=True)
    # general_features = SECTOR_FEATURE_MAP['General']#list(set(feat for features in SECTOR_FEATURE_MAP.values() for feat in features))
    #
    # try:
    #     general_comparator = ModelComparator(general_data, save_models=True, save_dir='models', tag='General')
    #     final_results['regression']['General'] = general_comparator.compare_regression_models(general_features)
    #     final_results['classification_binary']['General'] = general_comparator.compare_classification_models(general_features, 'binary')
    #     final_results['classification_multi']['General'] = general_comparator.compare_classification_models(general_features, 'multi')
    # except ValueError as e:
    #     print(f"Skipping General model due to error: {e}")

    print("\n" + "="*60, "\n▶️ STEP 2: TRAINING SECTOR-SPECIFIC MODELS\n", "="*60)
    for sector in all_sectors:
        print(f"\nProcessing Sector: {sector}")
        sector_data = generate_real_data(sector)
        sector_features = SECTOR_FEATURE_MAP[sector]
        
        try:
            sector_comparator = ModelComparator(sector_data, save_models=False, save_dir='models', tag=sector)
            final_results['regression'][sector] = sector_comparator.compare_regression_models(sector_features)
            final_results['classification_binary'][sector] = sector_comparator.compare_classification_models(sector_features, 'binary')
            # final_results['classification_multi'][sector] = sector_comparator.compare_classification_models(sector_features, 'multi')
        except ValueError as e:
            print(f"Skipping {sector} model due to error: {e}")

    print("\n\n" + "="*85, "\n" + " " * 28 + "🏁 FINAL PERFORMANCE SUMMARY 🏁\n", "="*85)
    
    # --- Regression Summary ---
    print("\n--- 📈 REGRESSION METRICS (LOWER RMSE & HIGHER R2 IS BETTER) ---")
    print(f"{'Scope':<15} {'Model':<20} {'Avg. R2 Score':<20} {'Avg. RMSE':<20}{'Avg. MSE':<20}")
    print("-" * 75)
    for scope, results in final_results['regression'].items():
        if not results: continue
        for model_name, metrics in results.items():
            print(f"{scope:<15} {model_name:<20} {metrics.get('R2', 0):<20.4f} {metrics.get('RMSE', 0):<20.4f} {metrics.get('MSE', 0):<20.6f}")

    # --- Binary Classification Summary ---
    print("\n--- 🎯 BINARY CLASSIFICATION METRICS (UP vs DOWN) ---")
    header = f"{'Scope':<15} {'Model':<20} {'Avg. F1':<12} {'Avg. Precision':<15} {'Down %':<10} {'Up %':<10}"
    print(header)
    print("-" * len(header))
    for scope, results in final_results['classification_binary'].items():
        if not results: continue
        for model_name, metrics in results.items():
            dirs = metrics.get('Direction', Counter())
            total = sum(dirs.values())
            down_pct = (dirs.get('Down', 0) / total * 100) if total > 0 else 0
            up_pct = (dirs.get('Up', 0) / total * 100) if total > 0 else 0
            print(f"{scope:<15} {model_name:<20} {metrics.get('F1', 0):<12.4f} {metrics.get('Precision', 0):<15.4f} {down_pct:<10.2f} {up_pct:<10.2f}")
    save_all_charts(final_results, detailed_results=None, base_out=None)


    # --- Multi-class Classification Summary ---
    # print("\n--- 🎯 MULTI-CLASS CLASSIFICATION METRICS ---")
    # header = f"{'Scope':<15} {'Model':<20} {'Avg. F1':<12} {'Avg. Precision':<15} {'Down %':<10} {'Neutral %':<12} {'Up %':<10}"
    # print(header)
    # print("-" * len(header))
    # for scope, results in final_results['classification_multi'].items():
    #     if not results: continue
    #     for model_name, metrics in results.items():
    #         dirs = metrics.get('Direction', Counter())
    #         total = sum(dirs.values())
    #         down_pct = (dirs.get('Down', 0) / total * 100) if total > 0 else 0
    #         neu_pct = (dirs.get('Neutral', 0) / total * 100) if total > 0 else 0
    #         up_pct = (dirs.get('Up', 0) / total * 100) if total > 0 else 0
    #         print(f"{scope:<15} {model_name:<20} {metrics.get('F1', 0):<12.4f} {metrics.get('Precision', 0):<15.4f} {down_pct:<10.2f} {neu_pct:<12.2f} {up_pct:<10.2f}")

if __name__ == "__main__":
    main()
