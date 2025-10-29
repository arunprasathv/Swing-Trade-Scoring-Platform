import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def compare_sectors():
    """Compare performance across different sectors."""
    # Sector ETFs to analyze
    sectors = {
        'XLK': 'Technology',
        'XLC': 'Communication Services',
        'XLY': 'Consumer Discretionary',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLP': 'Consumer Staples',
        'XLI': 'Industrials',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities'
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    
    sector_data = {}
    
    # Fetch data for each sector ETF
    for symbol in sectors.keys():
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-index columns
                data = pd.DataFrame({
                    'Open': data[('Open', symbol)] if ('Open', symbol) in data.columns else data['Open'],
                    'High': data[('High', symbol)] if ('High', symbol) in data.columns else data['High'],
                    'Low': data[('Low', symbol)] if ('Low', symbol) in data.columns else data['Low'],
                    'Close': data[('Close', symbol)] if ('Close', symbol) in data.columns else data['Close'],
                    'Volume': data[('Volume', symbol)] if ('Volume', symbol) in data.columns else data['Volume']
                })
            sector_data[symbol] = data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            
    # Calculate metrics for each sector
    sector_metrics = []
    
    for symbol, data in sector_data.items():
        try:
            # Calculate key metrics
            returns = data['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            sharpe = (returns.mean() * 252) / volatility  # Approximate Sharpe ratio
            
            # Calculate momentum metrics
            price = data['Close']
            ema9 = price.ewm(span=9, adjust=False).mean()
            ema21 = price.ewm(span=21, adjust=False).mean()
            ema50 = price.ewm(span=50, adjust=False).mean()
            
            last_close = price.iloc[-1]
            
            # Score based on EMA alignment
            ema_score = 0
            if ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
                ema_score = 2
            elif ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
                ema_score = -2
                
            # Score based on price vs EMAs
            price_score = 0
            if last_close > ema50.iloc[-1]:
                price_score += 1
            if last_close > ema21.iloc[-1]:
                price_score += 1
                
            # Calculate volume trend
            recent_volume = data['Volume'].tail(5)
            x = np.arange(len(recent_volume))
            volume_trend = np.polyfit(x, recent_volume.values, 1)[0]
            
            volume_score = 1 if volume_trend > 0 else -1
            
            total_score = ema_score + price_score + volume_score
            
            sector_metrics.append({
                'Sector': sectors[symbol],
                'Symbol': symbol,
                'Return (%)': returns.iloc[-1] * 100,
                'Volatility (%)': volatility * 100,
                'Sharpe Ratio': sharpe,
                'Technical Score': total_score,
                'Above EMA50': last_close > ema50.iloc[-1],
                'Volume Trend': 'Increasing' if volume_trend > 0 else 'Decreasing'
            })
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            
    # Convert to DataFrame and sort by technical score
    metrics_df = pd.DataFrame(sector_metrics)
    metrics_df = metrics_df.sort_values('Technical Score', ascending=False)
    
    return metrics_df
