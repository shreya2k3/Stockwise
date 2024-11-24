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

def load_stock_data(stock_data, intervals=['1d', '1h']):
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
            # Calculate indicators using ta library
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
        
    except Exception as e:
        print(f"Error fetching data for {symbol} at interval {interval}: {str(e)}")
        stock_data = None  # Ensure we return None to indicate failure

    return stock_data

import plotly.graph_objects as go
@st.cache_data(ttl=3600,show_spinner=False) 
def create_chart_for_indicator(stock_data, indicator_name, title, legend=True, hlines=None):
    """
    Creates an interactive figure for a specific stock indicator for displaying in Streamlit using Plotly.

    Parameters:
    - stock_data: DataFrame containing the stock data and indicators
    - indicator_name: list of column names in stock_data to plot
    - title: str, the title of the chart
    - legend: bool, whether to display the legend
    - hlines: list of tuples (y, color, dash_style) for horizontal lines

    Returns:
    - fig: The Plotly figure object to be displayed in Streamlit.
    """
    fig = go.Figure()

    # Plot each indicator as a separate line
    for name in indicator_name:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[name], mode='lines', name=name))

    # Add horizontal lines if specified
    if hlines:
        for hline in hlines:
            # Ensure the dash style is one of Plotly's accepted values
            # For a dashed line similar to Matplotlib's '--', use 'dash'
            dash_style = hline[2] if hline[2] in ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] else 'solid'
            
            fig.add_shape(type="line",
                          x0=stock_data.index.min(),
                          y0=hline[0],
                          x1=stock_data.index.max(),
                          y1=hline[0],
                          line=dict(color=hline[1], dash=dash_style),
                         )
            # Optionally add annotation next to the line
            fig.add_annotation(x=stock_data.index.mean(), y=hline[0], text=f"{hline[0]}", showarrow=False)

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", legend_title="Indicator", height =400)
    fig.update_xaxes(tickangle=-45)
    return fig

@st.cache_data(ttl=3600,show_spinner=False) 
def create_separate_charts(stock_data):
    """
    Generates separate charts for key stock indicators and displays them in two columns in Streamlit.
    
    Parameters:
    - stock_data: DataFrame containing the stock data and indicators
    """
    # Define configurations for different charts
    charts_config = [
        {'indicator_name': ['Adj Close', 'EMA_50', 'SMA_20'], 'title': 'Price Trend'},
        {'indicator_name': ['OBV_in_million'], 'title': 'On-Balance Volume (OBV) Indicator'},
        {'indicator_name': ['MACD', 'MACD_signal', 'MACD_histogram_12_26_9'], 'title': 'MACD Indicator'},
        {'indicator_name': ['RSI'], 'title': 'RSI Indicator', 'hlines': [(70, 'red', '--'), (30, 'green', '--')]},
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
    
    # If not found in environment variables, prompt the user to enter it
    if api_key is None:
        api_key = st.text_input("Enter your API key:", type="password")
        if api_key:
            st.success("API key entered successfully!")
    
    return api_key

@st.cache_data(ttl=3600,show_spinner=False) 
def run_openai(timeframe,symbol,last_day_summary):
    # st.session_state.api_key = st.secrets["api_key"]
      #st.session_state.openai_key = st.secrets["AI_KEY"]
    # client = Groq(api_key =st.session_state.api_key)
    api_key = get_api_key()
    
    if not api_key:
        st.error("No API key found. Please provide an API key to continue.")
        return None
    client = Groq(api_key=api_key)

    

    #latest_news = aisearch.serch_prompt_generate(symbol,search_mode=True)
    
    system_prompt = f"""
        Assume the role as a leading Technical Analysis (TA) expert in the crypto market,
        a modern counterpart to Charles Dow, John Bollinger, and Alan Andrews.
        Your mastery encompasses both stock and crypto fundamentals and intricate technical indicators.
        You possess the ability to decode complex market dynamics,
        providing clear insights and recommendations backed by a thorough understanding of interrelated factors.
        Your expertise extends to practical tools like the pandas_ta module,
        allowing you to navigate data intricacies with ease.
        As a TA authority, your role is to decipher market trends, make informed predictions, and offer valuable perspectives.
        Answer the following.
        1.What this given stock symbol represents?
        2. Given TA data as below on the last trading {timeframe}, what will be the next few {timeframe} possible stock price movement?
        3. Give me idea as both long term investment and short term trading.
        4. Given the final conclusion as Buy / Sell / Hold/ as in Confidence Level in % (give reason).
        5. Share summary of latest news on this ticker along with reference.
        5. Produce the result in markdown format : Analysis:, Conclusion & Recommendations:, Final Decision: Current Price = ,Long-term =  ,Short-term = ,Entry points = , Exit pints = ,Confidence level = %,News Summary .
        """
    response = client.chat.completions.create (
        #model = "gpt-4-1106-preview",
        model = "llama3-8b-8192",
        #model = 'gpt-3.5-turbo',
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""The token is {symbol} ,\nHere is the summary indicators.\n{last_day_summary}"""}],
        max_tokens=1024
    )
    ai_response = response.choices[0].message.content
    #wrapped_reply = textwrap.fill(reply, width=100)

    return ai_response


def streamlit_app(ticker_symbol):
    
    
    # with st.sidebar:
    #     st.write("## Built By")
    #     st.write("Name: Nyein Chan Ko Ko")
    #     st.write("GitHub: (https://github.com/nchanko)")
    col1,col2 = st.columns([1,3])
    with col1:
        st.image('App/StockWise_Logo.png',width=200)
    with col2:
        st.title("StockWise Investments")
        st.text("Here you will get in-depth analysis of different stocks!")
    st.markdown("*Some information on this page is AI-generated. This app is developed for educational purposes only and is not advisable to rely on it for financial decision-making.*")

    symbols = ['RELIANCE.NS','TATAMOTORS.NS','SJVN.NS', 'VEDANTA.NS', 'SAIL.NS', 'GAIL.NS','Custom symbol...']
    
    col1, col2, col3 = st.columns([4, 4, 2])
    with col1:
        st.write(f"Ticker Symbol: {ticker_symbol}")
        with st.container():
            selected_symbol = ticker_symbol
            st.write(f"Selected Symbol: {selected_symbol}")

            # selected_symbol = st.selectbox("Select a Symbol", symbols)

            # # Check if the user selected the option to enter a custom symbol
            # if selected_symbol == 'Custom symbol...':
            #     custom_symbol = st.text_input("Enter Custom Symbol")

            #     # Use the custom symbol if provided, otherwise keep showing the input field
            #     if custom_symbol:
            #         selected_symbol = custom_symbol

    with col2:
        interval_value = '1d'

    data_loaded = True
    # with col3:
    #     st.text("")
    #     st.text("")
    #     if st.button('Load Data', key='load_data'):
    #         data_loaded = True

    if data_loaded:
        with st.spinner(f'Loading data for {selected_symbol}...'):
            time.sleep(10)  # Simulate data loading
            
            # Assuming fetch_and_calculate_indicators & load_stock_data definitions are provided here
            # Simulate fetching and calculating indicators
            # stock_data = fetch_and_calculate_indicators(selected_symbol, interval_value)
            # summaries = load_stock_data(selected_symbol, stock_data, intervals=[interval_value])
            # Simulated placeholders for stock_data and summaries
            stock_data = fetch_and_calculate_indicators(selected_symbol, interval_value)
            # st.write(stock_data.columns)
            if stock_data is not None:
                summaries = load_stock_data(stock_data, intervals=[interval_value])

        textcol, chartcol = st.columns([4, 6])
        
        with textcol:
            if stock_data is not None:
                
                # Simulated AI response - replace with actual function call
                ai_response = run_openai(interval_value, selected_symbol, summaries)
                st.markdown(f"## *Our Verdict:*\n\n{ai_response}")
                st.markdown("*This analysis has been generated using AI and is intended solely for educational purposes. It is not advisable to rely on it for financial decision-making.*")
                #st.markdown(f"## *Summary for {interval_value} interval*")
            
                #st.json(summaries)

        with chartcol:
            if stock_data is not None:
                st.markdown(f"## *Stock Data for {selected_symbol}*")
                # Create separate charts function call should be placed here
                create_separate_charts(stock_data)

    st.write("---")
    st.write("#### About This App")
    st.write("""This demonstrates how to fetch stock data using yfinance, calculate technical indicators using pandas-ta, and display the data and indicators using Plotly in Streamlit.
                 \nBuilt by HackGoats""")
# Ensure the app runs only when directly executed
if __name__ == '__main__':
    ticker_symbol = st.query_params.get('symbol', 'AAPL')
    # st.write(f"Launching app with ticker symbol: {ticker_symbol}")
    streamlit_app(ticker_symbol)
