import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from groq import Groq
import sys
import os
from dotenv import load_dotenv
import ta

st.set_page_config(layout="wide")

def load_stock_data(stock_data, intervals=['1d']):
    summaries = {}
    for interval in intervals:
        if stock_data is not None and not stock_data.empty:
            # Generate a summary for the last available data point
            last_summary = stock_data.iloc[-1].to_dict()
            summary_formatted = {f"{key}_{interval}": value for key, value in last_summary.items()}
            summaries[interval] = summary_formatted
        else:
            summaries[interval] = {'error': f'No data available for {interval} interval.'}
    return summaries

@st.cache_data(ttl=3600) 
def fetch_and_calculate_indicators(symbol, interval):
    # Adjust the start date based on the interval
    end_date = datetime.today()
    if interval == '1d':
        start_date = end_date - timedelta(days=120)  # 4 months for daily data
    else:
        # Adjust for hourly and 15-minute data to ensure within the 60-day limit
        start_date = end_date - timedelta(days=60)  # Adjusted for yfinance limitation
    
    try:
        # Fetch stock data using yfinance with the specified interval
        stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
        
        if stock_data is not None and not stock_data.empty:
            # Ensure columns are properly flattened if they're MultiIndex
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Add Adj Close if it doesn't exist
            if 'Adj Close' not in stock_data.columns:
                stock_data['Adj Close'] = stock_data['Close']
            
            # Convert data to proper format and handle any remaining issues
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                if col in stock_data.columns:
                    # Ensure the column is 1-dimensional
                    if stock_data[col].ndim > 1:
                        stock_data[col] = stock_data[col].iloc[:, 0]
                    # Convert to numeric, replacing any non-numeric values with NaN
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            # Drop any rows with NaN values in essential columns
            stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            
            if stock_data.empty:
                st.error("No valid data available after cleaning.")
                return None
            
            # Calculate indicators using ta library
            try:
                # MACD
                macd = ta.trend.MACD(stock_data['Close'])
                stock_data['MACD'] = macd.macd()
                stock_data['MACD_signal'] = macd.macd_signal()
                stock_data['MACD_hist'] = macd.macd_diff()
                
                # RSI
                stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close']).rsi()
                
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(stock_data['Close'])
                stock_data['BBU'] = bb.bollinger_hband()
                stock_data['BBM'] = bb.bollinger_mavg()
                stock_data['BBL'] = bb.bollinger_lband()
                
                # OBV
                stock_data['OBV'] = ta.volume.OnBalanceVolumeIndicator(stock_data['Close'], stock_data['Volume']).on_balance_volume()
                
                # SMA and EMA
                stock_data['SMA_20'] = ta.trend.SMAIndicator(stock_data['Close'], window=20).sma_indicator()
                stock_data['EMA_50'] = ta.trend.EMAIndicator(stock_data['Close'], window=50).ema_indicator()
                
                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(stock_data['High'], stock_data['Low'], stock_data['Close'])
                stock_data['STOCH_K'] = stoch.stoch()
                stock_data['STOCH_D'] = stoch.stoch_signal()
                
                # ADX
                stock_data['ADX'] = ta.trend.ADXIndicator(stock_data['High'], stock_data['Low'], stock_data['Close']).adx()
                
                # Williams %R
                stock_data['WILLR'] = ta.momentum.WilliamsRIndicator(stock_data['High'], stock_data['Low'], stock_data['Close']).williams_r()
                
                # Chaikin Money Flow
                stock_data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume']).chaikin_money_flow()
                
                # Parabolic SAR
                stock_data['SAR'] = ta.trend.PSARIndicator(stock_data['High'], stock_data['Low'], stock_data['Close']).psar()
                
                # Convert OBV to million
                stock_data['OBV_in_million'] = stock_data['OBV'] / 1e6
                
                # Rename MACD histogram for consistency
                stock_data['MACD_histogram_12_26_9'] = stock_data['MACD_hist']
                
            except Exception as indicator_error:
                st.error(f"Error calculating technical indicators: {str(indicator_error)}")
                return None
        
    except Exception as e:
        print(f"Error fetching data for {symbol} at interval {interval}: {str(e)}")
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        stock_data = None  # Ensure we return None to indicate failure

    return stock_data

@st.cache_data(ttl=3600, show_spinner=False) 
def create_chart_for_indicator(stock_data, indicator_name, title, legend=True, hlines=None):
    """
    Creates an interactive figure for a specific stock indicator for displaying in Streamlit using Plotly.
    """
    fig = go.Figure()

    # Plot each indicator as a separate line
    for name in indicator_name:
        if name in stock_data.columns:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[name], mode='lines', name=name))

    # Add horizontal lines if specified
    if hlines:
        for hline in hlines:
            dash_style = hline[2] if hline[2] in ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] else 'solid'
            
            fig.add_shape(type="line",
                          x0=stock_data.index.min(),
                          y0=hline[0],
                          x1=stock_data.index.max(),
                          y1=hline[0],
                          line=dict(color=hline[1], dash=dash_style),
                         )
            fig.add_annotation(x=stock_data.index.mean(), y=hline[0], text=f"{hline[0]}", showarrow=False)

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", legend_title="Indicator", height=400)
    fig.update_xaxes(tickangle=-45)
    return fig

@st.cache_data(ttl=3600, show_spinner=False) 
def create_separate_charts(stock_data):
    """
    Generates separate charts for key stock indicators and displays them in Streamlit.
    """
    # Define configurations for different charts
    charts_config = [
        {'indicator_name': ['Adj Close', 'EMA_50', 'SMA_20'], 'title': 'Price Trend'},
        {'indicator_name': ['OBV_in_million'], 'title': 'On-Balance Volume (OBV) Indicator'},
        {'indicator_name': ['MACD', 'MACD_signal', 'MACD_histogram_12_26_9'], 'title': 'MACD Indicator'},
        {'indicator_name': ['RSI'], 'title': 'RSI Indicator', 'hlines': [(70, 'red', 'dash'), (30, 'green', 'dash')]},
        {'indicator_name': ['BBU', 'BBM', 'BBL', 'Adj Close'], 'title': 'Bollinger Bands'},
        {'indicator_name': ['STOCH_K', 'STOCH_D'], 'title': 'Stochastic Oscillator'},
        {'indicator_name': ['WILLR'], 'title': 'Williams %R'},
        {'indicator_name': ['ADX'], 'title': 'Average Directional Index (ADX)'},
        {'indicator_name': ['CMF'], 'title': 'Chaikin Money Flow (CMF)'}
    ]
    
    # Generate and display each chart directly in the app
    for config in charts_config:
        fig = create_chart_for_indicator(
            stock_data=stock_data,
            indicator_name=config['indicator_name'],
            title=config['title'],
            legend=True,
            hlines=config.get('hlines', None)
        )
        st.plotly_chart(fig, use_container_width=True)

def get_api_key():
    # First, try to get the API key from environment variables
    api_key = os.getenv("API_KEY")
    
    # If not found in environment variables, try Streamlit secrets
    if api_key is None:
        try:
            api_key = st.secrets.get("API_KEY") or st.secrets.get("api_key")
        except:
            pass
    
    # If still not found, prompt the user to enter it
    if api_key is None:
        api_key = st.text_input("Enter your Groq API key:", type="password")
        if api_key:
            st.success("API key entered successfully!")
    
    return api_key

@st.cache_data(ttl=3600, show_spinner=False) 
def run_openai(timeframe, symbol, last_day_summary):
    api_key = get_api_key()
    
    if not api_key:
        st.error("No API key found. Please provide a Groq API key to continue.")
        return "API key required for analysis. Please provide your Groq API key."
    
    try:
        client = Groq(api_key=api_key)
        
        system_prompt = f"""
        Assume the role as a leading Technical Analysis (TA) expert in the stock market,
        a modern counterpart to Charles Dow, John Bollinger, and Alan Andrews.
        Your mastery encompasses both stock and crypto fundamentals and intricate technical indicators.
        You possess the ability to decode complex market dynamics,
        providing clear insights and recommendations backed by a thorough understanding of interrelated factors.
        
        Answer the following:
        1. What this given stock symbol represents?
        2. Given TA data as below on the last trading {timeframe}, what will be the next few {timeframe} possible stock price movement?
        3. Give me idea as both long term investment and short term trading.
        4. Give the final conclusion as Buy / Sell / Hold with Confidence Level in % (give reason).
        5. Produce the result in markdown format: Analysis:, Conclusion & Recommendations:, Final Decision: Current Price = ,Long-term = ,Short-term = ,Entry points = , Exit points = ,Confidence level = %.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"The token is {symbol},\nHere is the summary indicators.\n{last_day_summary}"}
            ],
            max_tokens=1024
        )
        ai_response = response.choices[0].message.content
        return ai_response
    
    except Exception as e:
        st.error(f"Error getting AI analysis: {str(e)}")
        return "Error occurred while generating analysis. Please check your API key and try again."

def streamlit_app(ticker_symbol):
    # Header section
    col1, col2 = st.columns([1, 3])
    with col1:
        # Use a placeholder image if the original doesn't exist
        try:
            st.image('App/StockWise_Logo.png', width=200)
        except:
            st.write("📈 *StockWise*")
    
    with col2:
        st.title("StockWise Investments")
        st.text("Here you will get in-depth analysis of different stocks!")
    
    st.markdown("Some information on this page is AI-generated. This app is developed for educational purposes only and is not advisable to rely on it for financial decision-making.")

    # Symbol selection section
    col1, col2, col3 = st.columns([4, 4, 2])
    with col1:
        st.write(f"*Ticker Symbol:* {ticker_symbol}")
        st.write(f"*Selected Symbol:* {ticker_symbol}")

    with col2:
        interval_value = '1d'
        st.write(f"*Interval:* {interval_value}")

    # Add a manual load button for debugging
    with col3:
        if st.button('🔄 Refresh Data', key='load_data'):
            st.cache_data.clear()  # Clear cache to force refresh

    # Main data loading and display section
    with st.spinner(f'Loading data for {ticker_symbol}...'):
        # Fetch and calculate indicators
        stock_data = fetch_and_calculate_indicators(ticker_symbol, interval_value)
        
        # Debug information (you can remove this later)
        if stock_data is not None:
            st.success(f"Successfully loaded {len(stock_data)} rows of data")
            # st.write("Data shape:", stock_data.shape)  # Uncomment for debugging
            # st.write("Columns:", list(stock_data.columns))  # Uncomment for debugging
        
        if stock_data is not None and not stock_data.empty:
            summaries = load_stock_data(stock_data, intervals=[interval_value])
            
            # Create two columns for analysis and charts
            textcol, chartcol = st.columns([4, 6])
            
            with textcol:
                st.markdown("## Our Verdict:")
                # Get AI analysis
                ai_response = run_openai(interval_value, ticker_symbol, summaries)
                st.markdown(ai_response)
                st.markdown("This analysis has been generated using AI and is intended solely for educational purposes. It is not advisable to rely on it for financial decision-making.")

            with chartcol:
                st.markdown(f"## Stock Data for {ticker_symbol}")
                # Create and display charts
                create_separate_charts(stock_data)
                
        else:
            st.error(f"Failed to load data for {ticker_symbol}. Please check the symbol and try again.")
            st.info("Common issues:")
            st.info("• Check if the stock symbol is correct")
            st.info("• Ensure you have internet connection")
            st.info("• The market might be closed")

    # Footer
    st.write("---")
    st.write("#### About This App")
    st.write("""This demonstrates how to fetch stock data using yfinance, calculate technical indicators using pandas-ta, and display the data and indicators using Plotly in Streamlit.
                 \nBuilt by HackGoats""")

# Main execution
if __name__ == '__main__':
    # Get ticker symbol from query params or use default
    ticker_symbol = st.query_params.get('symbol', 'TATAMOTORS.NS')
    streamlit_app(ticker_symbol)