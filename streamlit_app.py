import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
import plotly.graph_objs as go
from swing_scoring_engine import analyze_ticker
from compare_scores import compare_scores, plot_score_comparison, plot_risk_reward_comparison

def init_session_state():
    """Initialize session state variables."""
    if 'results' not in st.session_state:
        st.session_state.results = pd.DataFrame()

def main():
    st.set_page_config(page_title="Swing Trading Scanner", layout="wide")
    init_session_state()
    
    # Organization name in sidebar
    st.sidebar.image("https://place-hold.it/300x100/1B243E/FFFFFF&text=Market%20Compass&bold&fontsize=20", use_column_width=True)
    st.sidebar.markdown("---")
    
    # Main title
    st.markdown("""
    <h1 style='text-align: center'>
        🚀 Swing Trade Scanner
    </h1>
    """, unsafe_allow_html=True)
    
    # Version info in sidebar
    st.sidebar.markdown("### Version: 1.0.0")
    st.sidebar.markdown("""
    Powered by Market Compass Technology
    
    ---
    """)
    
    # Main app interface
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("🎯 Scanner Settings")
        default_tickers = ("AAPL,MSFT,NVDA,AMD,GOOGL,META,AMZN,NFLX,TSLA,"
                         "JPM,BAC,GS,V,MA,WFC,"
                         "XOM,CVX,COP,"
                         "JNJ,UNH,PFE,ABBV,LLY,"
                         "PG,KO,PEP,COST,WMT,"
                         "CAT,BA,HON,GE,MMM,"
                         "HD,MCD,NKE,SBUX,DIS")
        
        use_default = st.checkbox("Use Default Stock List", value=True)
        if use_default:
            ticker_input = default_tickers
        else:
            ticker_input = st.text_input("Enter Tickers (comma-separated)", default_tickers)
        
        scan_button = st.button("Run Analysis")
        
        if scan_button:
            tickers = [t.strip().upper() for t in ticker_input.split(",")]
            results = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                try:
                    st.write(f"Analyzing {ticker}...")
                    result = analyze_ticker(ticker)
                    results.append(result)
                except Exception as e:
                    st.error(f"Error analyzing {ticker}: {str(e)}")
                progress_bar.progress((i + 1) / len(tickers))
            
            if results:
                df = pd.DataFrame(results)
                st.session_state.results = df
                
                st.success("Analysis complete! Expand results below to view details.")
    
    with col2:
        st.subheader("🧭 Market Context")
        if not st.session_state.results.empty:
            df = st.session_state.results
            spy_score = float(df['SPY/SPX Momentum Score'].iloc[0])
            
            # Display SPY score with color coding
            spy_color = "🟢" if spy_score >= 7 else "🟡" if spy_score >= 5 else "🔴"
            st.markdown(f"### S&P 500 Momentum Score: {spy_color} {spy_score:.1f}/10")
            st.markdown("""
            Score interpretation:
            - 7-10: Strong bullish momentum
            - 5-7: Moderate momentum
            - 0-5: Weak or bearish momentum
            """)
            
            st.markdown("---")
            st.subheader("📊 Stock Analysis Results")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_score = st.slider("Min Technical Score", 0.0, 10.0, 0.0)
            with col2:
                min_rr = st.slider("Min Risk/Reward Ratio", 0.0, 5.0, 0.0, step=0.5)
            with col3:
                # Initialize pattern filter options
                pattern_options = ['All']
                if not df.empty and 'Pattern' in df.columns:
                    pattern_options.extend(list(df['Pattern'].unique()))
                
                pattern_filter = st.multiselect(
                    "Pattern Filter",
                    options=pattern_options,
                    default=['All']
                )
            
            # Add price vs entry zone status
            def get_entry_status(row):
                current_price = float(row['Price'])
                entry_zone = row['Entry Zone'].split('–')
                entry_low = float(entry_zone[0])
                entry_high = float(entry_zone[1])
                if current_price < entry_low:
                    return "Below Entry"
                elif entry_low <= current_price <= entry_high:
                    return "In Entry Zone ✅"
                else:
                    return "Above Entry"

            df['Entry Status'] = df.apply(get_entry_status, axis=1)
            
            # Filter the dataframe
            filtered_df = df[
                (df['Ticker Technical Score'].astype(float) >= min_score) &
                (df['R:R (to TP1)'].astype(float) >= min_rr)
            ]

            # Additional entry zone filter
            entry_filter = st.radio(
                "Entry Zone Filter",
                ["All", "In Entry Zone", "Below Entry", "Above Entry"],
                horizontal=True
            )
            if entry_filter != "All":
                filtered_df = filtered_df[filtered_df['Entry Status'] == entry_filter]
            if 'All' not in pattern_filter:
                filtered_df = filtered_df[filtered_df['Pattern'].isin(pattern_filter)]
            
            # Display summary
            st.markdown(f"**Found {len(filtered_df)} stocks matching criteria**")
            
            # Sort by technical score if min_score > 0
            if min_score > 0:
                filtered_df = filtered_df.sort_values('Ticker Technical Score', ascending=False)
            
            # Display results for each stock
            for _, row in filtered_df.iterrows():
                with st.expander(f"📊 {row['Ticker']} - Score: {row['Ticker Technical Score']}/10"):
                    # Technical Score explanation
                    score_color = "🟢" if float(row['Ticker Technical Score']) >= 7 else "🟡" if float(row['Ticker Technical Score']) >= 5 else "🔴"
                    st.markdown(f"**Technical Score:** {score_color} {row['Ticker Technical Score']}/10 | **Current Price:** ${row['Price']}")
                    
                    st.markdown(f"**Pattern Detected:** {row['Pattern']}")
                    
                    # EMAs status
                    emas = row['EMA9/21/50'].split('/')
                    st.markdown("**Moving Averages:**")
                    st.markdown(f"- EMA9: ${emas[0]}")
                    st.markdown(f"- EMA21: ${emas[1]}")
                    st.markdown(f"- EMA50: ${emas[2]}")
                    
                    # RSI status
                    rsi = float(row['RSI(14)'])
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.markdown(f"**RSI(14):** {rsi:.1f} - {rsi_status}")
                    
                    # Trade setup
                    st.markdown("**Potential Trade Setup:**")
                    st.markdown(f"- Entry Zone: {row['Entry Zone']}")
                    st.markdown(f"- Target 1: {row['TP1/TP2'].split('/')[0]}")
                    st.markdown(f"- Stop Loss: {row['Stop Loss']}")
                    st.markdown(f"- Risk/Reward: {row['R:R (to TP1)']}")
            
            # Add comparison charts
            st.markdown("---")
            st.subheader("📈 Comparative Analysis")
            
            fig_scores = plot_score_comparison(filtered_df)
            st.plotly_chart(fig_scores, use_container_width=True)
            
            fig_rr = plot_risk_reward_comparison(filtered_df)
            st.plotly_chart(fig_rr, use_container_width=True)
            
            # Add export button
            st.markdown("---")
            if st.button("📥 Export Analysis"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scan_results_{timestamp}.csv"
                os.makedirs("output", exist_ok=True)
                filtered_df.to_csv(f"output/{filename}", index=False)
                st.success(f"Analysis exported to: {filename}")
        else:
            st.info("Enter tickers and click 'Run Analysis' to see results")

if __name__ == "__main__":
    main()
