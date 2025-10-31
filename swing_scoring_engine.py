#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trade Scoring Engine (US Equities) — Macro-Aware, SPY/Sector-Aligned
----------------------------------------------------------------------------
Data source: yfinance (daily/EOD). Designed for 2–10 day swing setups.

Adds: CSV export to ./output/swing_scores_YYYY-MM-DD_HHMM.csv
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_LOOKBACK_DAYS = 180  # ~6 months
EMA_PERIODS = (9, 21, 50)
RSI_PERIOD = 14

TICKER_TO_SECTOR_ETF: Dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "AVGO": "XLK", "CRM": "XLK",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF", "SCHW": "XLF",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    "UNH": "XLV", "JNJ": "XLV", "PFE": "XLV", "LLY": "XLV", "ABBV": "XLV",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "COST": "XLP", "WMT": "XLP",
    "CAT": "XLI", "BA": "XLI", "HON": "XLI", "GE": "XLI", "LMT": "XLI",
    "NEM": "XLB", "FCX": "XLB", "LIN": "XLB", "DOW": "XLB",
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "AEP": "XLU",
    "PLD": "XLRE", "AMT": "XLRE", "O": "XLRE", "SPG": "XLRE", "EQIX": "XLRE",
    "GOOGL": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC"
}

SECTOR_NAME_TO_SPIDER = {
    "Technology": "XLK", "Financial Services": "XLF", "Financial": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Health Care": "XLV", "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY", "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Industrials": "XLI", "Basic Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

VIX_SYMBOL = "^VIX"
TNX_SYMBOL = "^TNX"
DXY_SYMBOL = "DX-Y.NYB"

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def pct_change(series: pd.Series, n: int) -> float:
    if len(series) < n + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-1 - n]) - 1.0)

def fetch_history(tickers: List[str], days: int) -> Dict[str, pd.DataFrame]:
    unique = list(dict.fromkeys([t.upper() for t in tickers]))
    df = yf.download(unique, period=f"{days}d", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    data = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in unique:
            if t in df.columns.get_level_values(0):
                sub = df[t].copy()
                sub.dropna(how="all", inplace=True)
                data[t] = sub
    else:
        data[unique[0]] = df.copy()
    return data

def infer_sector_etf(ticker: str) -> str:
    t = ticker.upper()
    if t in TICKER_TO_SECTOR_ETF:
        return TICKER_TO_SECTOR_ETF[t]
    try:
        info = yf.Ticker(t).get_info()
        sector_name = info.get("sector") or info.get("industry") or ""
        for key, etf in SECTOR_NAME_TO_SPIDER.items():
            if key.lower() in str(sector_name).lower():
                return etf
    except Exception:
        pass
    return "XLK"

from dataclasses import dataclass

@dataclass
class ScoreComponents:
    ema: int
    rsi: int
    volume: int
    other: int = 0

def compute_scores_for_symbol(df: pd.DataFrame) -> ScoreComponents:
    e9 = ema(df["Close"], 9)
    e21 = ema(df["Close"], 21)
    e50 = ema(df["Close"], 50)
    r = rsi(df["Close"], 14)
    ema_score = 2 if (e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]) else (-2 if (e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]) else 0)
    r_last = float(r.iloc[-1])
    rsi_score = 2 if r_last > 60 else (-2 if r_last < 40 else 0)
    vol_trend = pct_change(df["Volume"], 5)
    volume_score = 2 if vol_trend > 0 else (-2 if vol_trend < 0 else 0)
    return ScoreComponents(ema=ema_score, rsi=rsi_score, volume=volume_score)

def normalize_to_10(raw_sum: int, min_sum: int = -8, max_sum: int = 8) -> float:
    clipped = max(min(raw_sum, max_sum), min_sum)
    return ((clipped - min_sum) / (max_sum - min_sum)) * 10.0

def score_spy(spy_df: pd.DataFrame, vix_df: Optional[pd.DataFrame]) -> float:
    comps = compute_scores_for_symbol(spy_df)
    vix_score = 0
    try:
        if vix_df is not None and len(vix_df) > 0:
            vix_last = float(vix_df["Close"].iloc[-1])
            vix_score = 2 if vix_last < 15 else (-2 if vix_last > 20 else 0)
    except Exception:
        pass
    raw = comps.ema + comps.rsi + comps.volume + vix_score
    return round(normalize_to_10(raw), 2)

def score_sector(sector_df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
    comps = compute_scores_for_symbol(sector_df)
    rel = sector_df["Close"] / spy_df["Close"].reindex_like(sector_df).ffill()
    rel_slope = rel.tail(20).iloc[-1] - rel.tail(20).iloc[0]
    rel_score = 2 if rel_slope > 0 else (-2 if rel_slope < 0 else 0)
    raw = comps.ema + comps.rsi + comps.volume + rel_score
    return round(normalize_to_10(raw), 2)

def score_macro(tnx_df: Optional[pd.DataFrame], dxy_df: Optional[pd.DataFrame], vix_df: Optional[pd.DataFrame], sector_etf: str) -> float:
    rate_score = 0; usd_score = 0; vix_score = 0
    try:
        if tnx_df is not None and len(tnx_df) > 5:
            tnx_change = pct_change(tnx_df["Close"], 5)
            if sector_etf in ("XLK", "XLC", "XLY"):
                rate_score = 2 if tnx_change < 0 else (-2 if tnx_change > 0 else 0)
            elif sector_etf in ("XLF", "XLE", "XLI", "XLB"):
                rate_score = 2 if tnx_change > 0 else (-2 if tnx_change < 0 else 0)
    except Exception:
        pass
    try:
        if dxy_df is not None and len(dxy_df) > 5:
            dxy_change = pct_change(dxy_df["Close"], 5)
            usd_score = 2 if dxy_change < 0 else (-2 if dxy_change > 0 else 0)
    except Exception:
        pass
    try:
        if vix_df is not None and len(vix_df) > 0:
            vix_last = float(vix_df["Close"].iloc[-1])
            vix_score = 2 if vix_last < 15 else (-2 if vix_last > 20 else 0)
    except Exception:
        pass
    raw = rate_score + usd_score + vix_score + 4
    return float(max(0, min(10, raw)))

def score_ticker(ticker_df: pd.DataFrame):
    comps = compute_scores_for_symbol(ticker_df)
    last = float(ticker_df["Close"].iloc[-1])
    e9 = float(ema(ticker_df["Close"], 9).iloc[-1])
    e21 = float(ema(ticker_df["Close"], 21).iloc[-1])
    e50 = float(ema(ticker_df["Close"], 50).iloc[-1])
    structure_score = 2 if (last > e21 and e21 > e50) else (-2 if (last < e21 and e21 < e50) else 0)
    raw = comps.ema + comps.rsi + comps.volume + structure_score
    return round(normalize_to_10(raw), 2), {"last": last, "ema9": e9, "ema21": e21, "ema50": e50}

def detect_consolidation(df: pd.DataFrame, window: int = 20) -> bool:
    """Detect if price is in consolidation pattern."""
    recent = df.tail(window)
    atr = (recent['High'] - recent['Low']).mean()
    price_range = recent['High'].max() - recent['Low'].min()
    return price_range <= (atr * 1.5)

def detect_resistance_test(df: pd.DataFrame, window: int = 20) -> bool:
    """Detect if price is testing resistance."""
    recent = df.tail(window)
    current = float(recent['Close'].iloc[-1])
    recent_high = float(recent['High'].max())
    return abs(current - recent_high) / recent_high < 0.02

def classify_strategy(df: pd.DataFrame, last, e9, e21, e50, rsi_val, recent_high, recent_low) -> str:
    # Check for consolidation first
    if detect_consolidation(df):
        if last > e50:
            return "Consolidation (bullish)"
        elif last < e50:
            return "Consolidation (bearish)"
        return "Consolidation (neutral)"
        
    # Check for resistance test
    if detect_resistance_test(df):
        if e9 > e21 and rsi_val > 50:
            return "Testing Resistance (bullish)"
        return "Testing Resistance (weak)"
    
    # Momentum patterns
    if (e9 > e21 > e50) and rsi_val > 60 and last > recent_high:
        return "Momentum / Breakout"
    if (e9 < e21 < e50) and rsi_val < 40 and last < recent_low:
        return "Momentum / Breakdown"
        
    # Mean reversion patterns
    if rsi_val < 35 and e21 <= last <= e50:
        return "Mean Reversion (oversold bounce)"
    if rsi_val > 65 and e50 <= last <= e21:
        return "Mean Reversion (overbought pullback)"
        
    # Reversal patterns
    if last > e50 and e9 > e21 and rsi_val > 55:
        if last > recent_high:
            return "Bullish Reversal (confirmed)"
        return "Bullish Reversal (potential)"
    if last < e50 and e9 < e21 and rsi_val < 45:
        if last < recent_low:
            return "Bearish Reversal (confirmed)"
        return "Bearish Reversal (potential)"
        
    return "Neutral / Wait"

def compute_wcs(spy_score: float, sector_score: float, macro_score: float, ticker_score: float) -> float:
    return round(0.35 * spy_score + 0.25 * sector_score + 0.20 * macro_score + 0.20 * ticker_score, 2)

def compute_success_probability(wcs: float, spy_score: float, sector_score: float, macro_score: float) -> float:
    base = wcs * 9.5
    boost = 3.0 if (spy_score >= 8 and sector_score >= 8 and macro_score >= 8) else 0.0
    penalty = 5.0 if (spy_score < 5 or macro_score < 5) else 0.0
    prob = max(0.0, min(100.0, base + boost - penalty))
    return round(prob, 1)

def suggest_levels(df: pd.DataFrame):
    """Calculate trade levels with tighter risk management."""
    close = df["Close"]
    current_close = float(close.iloc[-1])
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    e21 = ema(close, 21).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    
    # Use recent swing points (5-day window for tighter control)
    recent_low = float(df["Low"].tail(5).min())
    recent_high = float(df["High"].tail(5).max())
    
    # Entry zone around EMA21
    entry_low = float(e21 - 0.3 * atr)
    entry_high = float(e21 + 0.3 * atr)
    
    # Targets based on ATR and recent price action
    r1 = abs(current_close - recent_low)  # Recent range
    tp1 = float(current_close + r1)  # First target at recent range
    tp2 = float(tp1 + atr)  # Second target one ATR beyond first
    
    # Tighter stop loss: recent low or nearest EMA, no additional ATR buffer
    stop_candidates = [
        recent_low,  # Recent swing low
        e21 if current_close > e21 else float('-inf'),  # EMA21 if above it
        e50 if current_close > e50 else float('-inf')  # EMA50 if above it
    ]
    stop = max([s for s in stop_candidates if s != float('-inf')])
    
    return (round(entry_low, 2), round(entry_high, 2)), (round(tp1, 2), round(tp2, 2)), round(stop, 2)

def analyze_ticker(ticker: str, days: int = DEFAULT_LOOKBACK_DAYS):
    ticker = ticker.upper()
    sector_etf = infer_sector_etf(ticker)
    symbols = ["SPY", sector_etf, ticker, VIX_SYMBOL, TNX_SYMBOL, DXY_SYMBOL]
    data = fetch_history(symbols, days=days)
    if "SPY" not in data or ticker not in data or sector_etf not in data:
        raise RuntimeError("Missing data for SPY, Sector ETF, or Ticker.")
    spy_df = data["SPY"].dropna()
    sector_df = data[sector_etf].dropna()
    ticker_df = data[ticker].dropna()
    vix_df = data.get(VIX_SYMBOL, pd.DataFrame()).dropna()
    tnx_df = data.get(TNX_SYMBOL, pd.DataFrame()).dropna()
    dxy_df = data.get(DXY_SYMBOL, pd.DataFrame()).dropna()
    spy_score = score_spy(spy_df, vix_df)
    sector_score = score_sector(sector_df, spy_df)
    macro_score = score_macro(tnx_df, dxy_df, vix_df, sector_etf)
    ticker_score, levels = score_ticker(ticker_df)
    rsi_val = float(rsi(ticker_df["Close"]).iloc[-1])
    recent_high = float(ticker_df["Close"].tail(20).max())
    recent_low = float(ticker_df["Close"].tail(20).min())
    strat = classify_strategy(ticker_df, levels["last"], levels["ema9"], levels["ema21"], levels["ema50"], rsi_val, recent_high, recent_low)
    wcs = compute_wcs(spy_score, sector_score, macro_score, ticker_score)
    prob = compute_success_probability(wcs, spy_score, sector_score, macro_score)
    entry_range, tps, stop = suggest_levels(ticker_df)
    current_price = levels["last"]
    
    # Calculate R:R using current price instead of entry mid
    reward = tps[0] - current_price  # Distance to first target
    risk = current_price - stop  # Distance to stop
    
    if risk <= 0 or reward <= 0:  # Invalid setup
        rr = 0
    else:
        # Cap risk/reward at 5:1 to avoid unrealistic ratios from tiny stops
        risk_pct = risk / current_price  # Risk as percentage of price
        if risk_pct < 0.002:  # If risk is less than 0.2% of price
            risk = current_price * 0.002  # Use minimum 0.2% risk
        rr = round(min(reward / risk, 5.0), 2)  # Cap at 5:1
    # Calculate ATR for display
    atr = round(float((ticker_df["High"] - ticker_df["Low"]).rolling(14).mean().iloc[-1]), 2)
    
    trade_rating = round(wcs, 1)
    return {
        "Ticker": ticker,
        "Sector ETF": sector_etf,
        "SPY/SPX Momentum Score": spy_score,
        "Sector Strength Score": sector_score,
        "Macro Context Score": macro_score,
        "Ticker Technical Score": ticker_score,
        "Weighted Conviction Score": wcs,
        "Trade Rating (/10)": trade_rating,
        "Success Probability (%)": prob,
        "Strategy Type": strat,
        "Entry Zone": f"{entry_range[0]} – {entry_range[1]}",
        "TP1/TP2": f"{tps[0]} / {tps[1]}",
        "Stop Loss": stop,
        "R:R (to TP1)": rr,
        "RSI(14)": round(rsi_val, 1),
        "ATR(14)": atr,
        "EMA9/21/50": f"{round(levels['ema9'],2)} / {round(levels['ema21'],2)} / {round(levels['ema50'],2)}",
        "Price": round(levels["last"], 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Swing Trade Scoring Engine (US Equities, EOD via yfinance)")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of tickers, e.g. NVDA AAPL JPM")
    parser.add_argument("--days", type=int, default=DEFAULT_LOOKBACK_DAYS, help=f"History length in days (default {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--csv", action="store_true", help="Export results to CSV in ./output")
    args = parser.parse_args()

    rows = []
    for t in args.tickers:
        try:
            res = analyze_ticker(t, days=args.days)
            rows.append(res)
        except Exception as e:
            rows.append({"Ticker": t.upper(), "Error": str(e)})

    # Convert to DataFrame and sort by Technical Score (descending)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Ticker Technical Score", "R:R (to TP1)", "Success Probability (%)"], 
                       ascending=[False, False, False])
    
    # Display sorted results
    with pd.option_context("display.max_columns", None):
        print("\nResults (Sorted by Technical Score → Risk/Reward → Success Probability):")
        print(df.to_string(index=False))

    if args.csv:
        os.makedirs("output", exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M")
        path = os.path.join("output", f"swing_scores_{ts}.csv")
        df.to_csv(path, index=False)
        print(f"\nCSV saved → {path}")

if __name__ == "__main__":
    main()
