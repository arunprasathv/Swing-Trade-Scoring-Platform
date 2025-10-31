import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from swing_scoring_engine import analyze_ticker

# ---- UI Configuration ----
st.set_page_config(page_title="Stratify™", page_icon="📈", layout="wide")

# ---- Header ----
col1, col2, col3 = st.columns([2,1,2])
with col2:
    st.image("assets/stratify.png", width=150, use_column_width=False)

# Add separator
st.markdown("<hr style='margin: 0 0 30px 0; opacity: 0.2;'>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
    .stButton > button {
        background-color: #28a745;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        margin: 0;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .company-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #1E1E1E;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 24px;
        text-align: center;
        color: #4A4A4A;
        margin-top: 0;
    }
    footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# ---- Market Context (Always Visible) ----
try:
    # Get SPY data
    spy = yf.Ticker("SPY")
    spy_hist = spy.history(period="2d")
    prev_close = float(spy_hist['Close'].iloc[-2])
    spy_price = float(spy_hist['Close'].iloc[-1])
    price_change = spy_price - prev_close
    
    # Get analysis
    spy_analysis = analyze_ticker("SPY")
    spy_score = spy_analysis["SPY/SPX Momentum Score"]
    
    # Calculate and format change
    delta = f"{'+' if price_change > 0 else ''}{price_change:.2f}"
    delta_color = "normal" if price_change > 0 else "inverse"
    
    # Display market context
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("🌎 Market Score", f"{spy_score}/10", "SPY/SPX")
    with col2:
        st.metric("📊 SPY Price", f"${spy_price:.2f}", delta, delta_color=delta_color)
        
except Exception as e:
    st.error(f"Error getting market context: {str(e)}")

# ---- Input Section ----
input_container = st.container()
with input_container:
    col1, col2, col3, col4 = st.columns([3,2,2,1])
    with col1:
        tickers_input = st.text_input("📝 Enter Tickers (comma-separated)", value="MSFT, NVDA, AAPL")
    with col2:
        min_tech_score = st.slider("📊 Min Technical Score", 0.0, 10.0, 5.0, 0.5)
    with col3:
        min_rr = st.slider("⚖️ Min Risk/Reward", 0.0, 3.0, 1.0, 0.1)
    with col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        scan_button = st.button("🔍 Scan")
        
if scan_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    progress = st.progress(0)
    
    results = []
    for idx, ticker in enumerate(tickers, 1):
        try:
            analysis = analyze_ticker(ticker)
            
            # Only include if meets minimum criteria
            tech_score = float(analysis["Ticker Technical Score"])
            rr = float(analysis["R:R (to TP1)"])
            
            if tech_score >= min_tech_score and rr >= min_rr:
                results.append(analysis)
                
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            
        progress.progress(idx/len(tickers))
    
    if results:
        # Convert to DataFrame and sort by multiple criteria
        df = pd.DataFrame(results)
        df = df.sort_values(
            by=["Ticker Technical Score", "R:R (to TP1)", "Success Probability (%)"],
            ascending=[False, False, False]
        )
        
        # Show sorting criteria
        st.info("📊 Results sorted by: Technical Score → Risk/Reward → Success Probability")
        
        # Display results in sections
        for _, row in df.iterrows():
            # Pattern-based color and icon
            pattern = row['Strategy Type']
            if "bullish" in pattern.lower():
                pattern_color = "🟢"
            elif "bearish" in pattern.lower():
                pattern_color = "🔴"
            elif "neutral" in pattern.lower() or "consolidation" in pattern.lower():
                pattern_color = "🟡"
            else:
                pattern_color = "⚪"
                
            # Get price change
            ticker = row['Ticker']
            current_price = row['Price']
            ticker_hist = yf.Ticker(ticker).history(period="2d")
            price_text = f"${current_price:.2f}"
            
            if len(ticker_hist) >= 2:
                prev_close = float(ticker_hist['Close'].iloc[-2])
                price_change = current_price - prev_close
                pct_change = (price_change / prev_close) * 100
                pattern_color = "🟢" if price_change > 0 else "🔴"
                
                sign = "+" if price_change > 0 else ""
                arrow = "↑" if price_change > 0 else "↓"
                price_text = f"${current_price:.2f} {arrow} ({sign}${abs(price_change):.2f}, {sign}{pct_change:.1f}%)"
            
            # Create header text
            header = f"{pattern_color} {row['Ticker']} {price_text} - {row['Strategy Type']} ({row['Ticker Technical Score']}/10)"
            
            # Create expander
            expander = st.expander(header)
            with expander:
                cols = st.columns(3)
                
                # Column 1: Pattern & Scores
                with cols[0]:
                    st.markdown("### 📈 Pattern Analysis")
                    
                    # Pattern classification
                    pattern_type = row['Strategy Type'].split()[0].lower()
                    if "consolidation" in pattern_type:
                        st.info(f"📦 {row['Strategy Type']}")
                    elif "momentum" in pattern_type:
                        st.success(f"🚀 {row['Strategy Type']}")
                    elif "reversal" in pattern_type:
                        st.warning(f"↩️ {row['Strategy Type']}")
                    elif "testing" in pattern_type:
                        st.info(f"🎯 {row['Strategy Type']}")
                    elif "mean" in pattern_type:
                        st.warning(f"↕️ {row['Strategy Type']}")
                    else:
                        st.write(f"⏳ {row['Strategy Type']}")
                        
                    st.metric("Technical Score", f"{row['Ticker Technical Score']}/10")
                    st.metric("Sector Score", f"{row['Sector Strength Score']}/10")
                
                # Column 2: Trade Setup
                with cols[1]:
                    st.markdown("### 🎯 Trade Setup")
                    
                    # Entry Zone with tooltip
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.write(f"Entry Zone: {row['Entry Zone']}")
                    with col2:
                        st.help("""
                        Entry Zone = EMA21 ± (0.3 × ATR)
                        • Calculated around 21-day moving average
                        • ±30% of ATR provides flexibility
                        • Best entry: lower end of range
                        • Waits for pullback to key support
                        """)
                    
                    # Stop Loss with tooltip
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.write(f"Stop Loss: {row['Stop Loss']}")
                    with col2:
                        st.help("""
                        Stop Loss = Recent swing low or nearest EMA
                        • Uses 5-day swing low
                        • Or EMA21/EMA50 if price is above them
                        • No ATR buffer added (tighter risk)
                        • Protects against breakdown
                        """)
                    
                    # Targets with tooltip
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.write(f"Targets: {row['TP1/TP2']}")
                    with col2:
                        st.help("""
                        TP1 = Current Price + Recent Range
                        TP2 = TP1 + ATR
                        
                        • TP1: Recovers recent 5-day decline
                        • TP2: Extended target with momentum
                        • Based on actual price behavior
                        • Volatility-adjusted using ATR
                        """)
                    
                    # Risk/Reward with tooltip
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.metric("Risk/Reward", str(row['R:R (to TP1)']))
                    with col2:
                        st.help("""
                        R:R = (TP1 - Current Price) / (Current Price - Stop)
                        
                        • Measures potential reward vs risk
                        • Minimum 1:1 for viable trades
                        • 2:1+ considered good setups
                        • Capped at 5:1 to avoid unrealistic ratios
                        """)
                
                # Column 3: Technical Details
                with cols[2]:
                    st.markdown("### 📊 Technical Details")
                    st.write(f"RSI(14): {row['RSI(14)']}")
                    st.write(f"ATR(14): ${row['ATR(14)']}")
                    
                    # Parse EMAs and display individually with price comparison
                    emas = row['EMA9/21/50'].split(" / ")
                    current_price = row['Price']
                    st.write("EMAs:")
                    
                    # Compare price with each EMA
                    ema9 = float(emas[0])
                    ema21 = float(emas[1])
                    ema50 = float(emas[2])
                    
                    st.write(f"• 9: ${ema9:.2f} {'✅' if current_price > ema9 else '❌'}")
                    st.write(f"• 21: ${ema21:.2f} {'✅' if current_price > ema21 else '❌'}")
                    st.write(f"• 50: ${ema50:.2f} {'✅' if current_price > ema50 else '❌'}")
                    
                    # Calculate volume metrics
                    hist = yf.Ticker(row['Ticker']).history(period="30d")
                    avg_vol_20d = hist['Volume'].iloc[-20:].mean()
                    current_vol = hist['Volume'].iloc[-1]
                    vol_ratio = (current_vol / avg_vol_20d) * 100
                    
                    # Format volume numbers
                    def format_volume(vol):
                        if vol >= 1_000_000:
                            return f"{vol/1_000_000:.1f}M"
                        elif vol >= 1_000:
                            return f"{vol/1_000:.1f}K"
                        return str(int(vol))
                    
                    st.write("")  # Add spacing
                    st.write("Volume Analysis:")
                    
                    # Volume display with scoring impact
                    if vol_ratio > 100:
                        vol_impact = "Bullish 📈" if price_change > 0 else "Bearish 📉"
                        st.write(f"• Today: {format_volume(current_vol)} (↑ {vol_ratio:.0f}% of avg)")
                        st.caption(f"Volume expansion with {vol_impact} price action")
                    else:
                        st.write(f"• Today: {format_volume(current_vol)} (↓ {vol_ratio:.0f}% of avg)")
                        st.caption("Volume below average - reduced conviction")
                        
                    st.write(f"• 20d Avg: {format_volume(avg_vol_20d)}")
                    
                    # Volume scoring explanation
                    st.write("")  # Add spacing
                    st.write("ℹ️ Volume Impact on Score:")
                    st.caption("""
                    • Above average volume (>100%) adds conviction
                    • Volume expansion with price trend adds score
                    • Low volume reduces pattern confidence
                    • Volume trend aligned with price trend increases score
                    """)
        
        # Add download button at bottom
        st.write("")  # Add spacing
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Analysis",
            csv,
            "stratify_analysis.csv",
            "text/csv",
            key='download-csv',
            help="Download detailed analysis in CSV format"
        )
        
    else:
        st.warning("No tickers met the minimum criteria.")

# ---- Footer ----
st.markdown("""
<footer>
    <p style='margin-bottom: 5px;'>Data source: yfinance (EOD) • Designed for 2-10 day swing setups</p>
    <p style='margin: 0;'><strong>Stratify™</strong> • Professional Trading Analysis</p>
</footer>
""", unsafe_allow_html=True)
