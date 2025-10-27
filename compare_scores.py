import pandas as pd
import plotly.graph_objs as go
from swing_scoring_engine import analyze_ticker

def compare_scores(tickers):
    """Compare technical scores and other metrics between multiple tickers."""
    results = []
    
    for ticker in tickers:
        try:
            result = analyze_ticker(ticker)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    # Convert string scores to float
    df['Ticker Technical Score'] = df['Ticker Technical Score'].astype(float)
    df['SPY/SPX Momentum Score'] = df['SPY/SPX Momentum Score'].astype(float)
    df['R:R (to TP1)'] = df['R:R (to TP1)'].astype(float)
    
    return df

def plot_score_comparison(df):
    """Create a bar plot comparing technical scores."""
    fig = go.Figure()
    
    # Add technical score bars
    fig.add_trace(go.Bar(
        x=df['Ticker'],
        y=df['Ticker Technical Score'],
        name='Technical Score',
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add SPY momentum score bars
    fig.add_trace(go.Bar(
        x=df['Ticker'],
        y=df['SPY/SPX Momentum Score'],
        name='SPY/SPX Momentum',
        marker_color='rgb(26, 118, 255)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Technical Score Comparison',
        xaxis_tickangle=-45,
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        template='plotly_dark'
    )
    
    return fig

def plot_risk_reward_comparison(df):
    """Create a bar plot comparing risk/reward ratios."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Ticker'],
        y=df['R:R (to TP1)'],
        name='Risk/Reward Ratio',
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.update_layout(
        title='Risk/Reward Ratio Comparison',
        xaxis_tickangle=-45,
        yaxis_title='R:R Ratio',
        template='plotly_dark'
    )
    
    return fig

def export_comparison(df, filename='comparison_results.csv'):
    """Export comparison results to CSV."""
    try:
        df.to_csv(f"output/{filename}", index=False)
        return True
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return False
