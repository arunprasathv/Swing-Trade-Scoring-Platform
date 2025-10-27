import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Sector ETF mappings
SECTOR_ETFS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL', 'META', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'AVGO', 'QCOM', 'INTC', 'IBM'],
    'XLC': ['GOOGL', 'META', 'NFLX', 'DIS', 'TMUS', 'CMCSA', 'VZ', 'T', 'EA', 'ATVI'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'TGT', 'BKNG'],
    'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'USB', 'AXP', 'V', 'MA', 'PYPL'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'PSX', 'VLO', 'MPC', 'KMI'],
    'XLV': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'DHR', 'BMY', 'AMGN'],
    'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'EL', 'CL', 'KMB'],
    'XLI': ['HON', 'UPS', 'CAT', 'DE', 'LMT', 'RTX', 'UNP', 'BA', 'MMM', 'GE'],
    'XLB': ['LIN', 'ECL', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC'],
    'XLRE': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'DLR', 'WELL', 'AVB', 'EQR'],
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'XEL', 'ED', 'EXC', 'WEC']
}

def get_sector_etf(ticker):
    """Find the sector ETF for a given ticker."""
    ticker = ticker.upper()
    for etf, components in SECTOR_ETFS.items():
        if ticker in components:
            return etf
    # Default to technology sector for newer tech companies
    return 'XLK'

def calculate_atr(data, period=14):
    """Calculate Average True Range."""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_ema_score(data):
    """Calculate score based on EMA alignments."""
    last_row = data.iloc[-1]
    ema_9 = last_row['EMA9']
    ema_21 = last_row['EMA21']
    ema_50 = last_row['EMA50']
    
    if ema_9 > ema_21 > ema_50:
        return 2
    elif ema_9 < ema_21 < ema_50:
        return -2
    return 0

def calculate_rsi_score(rsi):
    """Calculate score based on RSI value."""
    if rsi > 60:
        return 2
    elif rsi < 40:
        return -2
    return 0

def calculate_volume_score(data):
    """Calculate score based on volume trend."""
    recent_volume = data['Volume'].tail(5)
    volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
    
    if volume_trend > 0:
        return 2
    elif volume_trend < 0:
        return -2
    return 0

def detect_consolidation(data, atr_multiple=1.5):
    """Detect if price is in consolidation."""
    last_5_days = data.tail(5)
    atr = calculate_atr(data).iloc[-1]
    price_range = last_5_days['High'].max() - last_5_days['Low'].min()
    
    return price_range <= (atr * atr_multiple)

def calculate_risk_reward(entry, stop, target):
    """Calculate risk-reward ratio."""
    if stop >= entry:  # Invalid stop loss
        return 0
    
    risk = entry - stop
    reward = target - entry
    
    if risk == 0:  # Avoid division by zero
        return 0
        
    return reward / risk

def analyze_ticker(ticker):
    """Main analysis function for a ticker."""
    try:
        # Determine sector ETF
        sector_etf = get_sector_etf(ticker)
        print(f"Sector ETF for {ticker}: {sector_etf}")
        
        # Get data for SPY, sector ETF, and ticker
        symbols = ['SPY', sector_etf, ticker, '^VIX', '^TNX', 'DX-Y.NYB']
        print(f"Fetching data for symbols: {symbols}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)
        
        # First check cache
        cache_dir = "output"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = f"{cache_dir}/{ticker}_cache.csv"
        
        if os.path.exists(cache_file):
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if cache_age < timedelta(hours=4):  # Use cache if less than 4 hours old
                print("Using cached data")
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            else:
                print("Fetching fresh data")
                data = yf.download(ticker, start=start_date, end=end_date)
                data.to_csv(cache_file)
        else:
            print("Fetching fresh data")
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(cache_file)
            
        spy_data = yf.download('SPY', start=start_date, end=end_date)
        vix_data = yf.download('^VIX', start=start_date, end=end_date)
        
        print(f"Data fetched successfully for {len(symbols)} symbols")
        
        # Calculate technical indicators
        data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
        data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        
        # Calculate component scores
        ema_score = calculate_ema_score(data)
        rsi_score = calculate_rsi_score(data['RSI'].iloc[-1])
        volume_score = calculate_volume_score(data)
        
        # Calculate SPY score
        print("\nCalculating SPY score:")
        spy_ema_score = calculate_ema_score(spy_data)
        spy_rsi_score = calculate_rsi_score(calculate_rsi(spy_data['Close']))
        spy_volume_score = calculate_volume_score(spy_data)
        vix_score = 0 if 15 <= vix_data['Close'].iloc[-1] <= 25 else -2
        
        print(f"- EMA score: {spy_ema_score}")
        print(f"- RSI score: {spy_rsi_score}")
        print(f"- Volume score: {spy_volume_score}")
        print(f"- VIX last: {vix_data['Close'].iloc[-1]:.2f}, VIX score: {vix_score}")
        
        spy_raw_score = spy_ema_score + spy_rsi_score + spy_volume_score + vix_score
        spy_normalized_score = (spy_raw_score + 8) * 10/16
        
        print(f"Raw score: {spy_raw_score}")
        print(f"Normalized score: {spy_normalized_score}")
        
        # Calculate price levels
        current_price = data['Close'].iloc[-1]
        atr = calculate_atr(data).iloc[-1]
        
        entry_zone_low = current_price - (0.5 * atr)
        entry_zone_high = current_price + (0.5 * atr)
        stop_loss = current_price - (2 * atr)
        target_1 = current_price + (3 * atr)
        target_2 = current_price + (5 * atr)
        
        # Detect pattern
        is_consolidating = detect_consolidation(data)
        volume_increasing = volume_score > 0
        
        if is_consolidating and volume_increasing:
            pattern = "Consolidation with increasing volume"
        elif is_consolidating:
            pattern = "Consolidation"
        elif volume_increasing:
            pattern = "Volume expansion"
        else:
            pattern = "No clear pattern"
            
        # Calculate final technical score
        raw_score = ema_score + rsi_score + volume_score
        if data['Close'].iloc[-1] > data['EMA50'].iloc[-1]:
            raw_score += 2
        elif data['Close'].iloc[-1] < data['EMA50'].iloc[-1]:
            raw_score -= 2
            
        normalized_score = (raw_score + 8) * 10/16
        
        print(f"Analysis complete for {ticker}")
        
        return {
            'Ticker': ticker,
            'Sector ETF': sector_etf,
            'Price': f"{current_price:.2f}",
            'Pattern': pattern,
            'Entry Zone': f"{entry_zone_low:.2f}–{entry_zone_high:.2f}",
            'Stop Loss': f"{stop_loss:.2f}",
            'TP1/TP2': f"{target_1:.2f}/{target_2:.2f}",
            'R:R (to TP1)': f"{calculate_risk_reward(current_price, stop_loss, target_1):.2f}",
            'ATR': f"{atr:.2f}",
            'RSI(14)': f"{data['RSI'].iloc[-1]:.1f}",
            'EMA9/21/50': f"{data['EMA9'].iloc[-1]:.2f}/{data['EMA21'].iloc[-1]:.2f}/{data['EMA50'].iloc[-1]:.2f}",
            'Ticker Technical Score': f"{normalized_score:.1f}",
            'SPY/SPX Momentum Score': f"{spy_normalized_score:.1f}",
            'Strategy Type': 'Momentum' if normalized_score >= 7 else 'Mean Reversion' if normalized_score <= 3 else 'Mixed'
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        raise e

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return pd.Series(rsi, index=prices.index)
