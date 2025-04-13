import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json

# Set page configuration
st.set_page_config(layout="wide", page_title="Stock Price Predictor")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 1rem;
    }
    .subtitle {
        color: #17a2b8;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown('<h1 class="title">Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; color: #666;">Predict future stock prices with machine learning</p>', unsafe_allow_html=True)

# Function to search for companies
def search_company(query):
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'quotes' in data and len(data['quotes']) > 0:
            suggestions = []
            for quote in data['quotes']:
                if 'symbol' in quote and 'shortname' in quote:
                    suggestions.append({
                        'symbol': quote['symbol'],
                        'name': quote['shortname'],
                        'exchange': quote.get('exchange', 'N/A')
                    })
            return suggestions
        return []
    except Exception as e:
        st.error(f"Error searching for company: {str(e)}")
        return []

# Function to get currency symbol based on the stock exchange or symbol
def get_currency_symbol(ticker_obj, stock_symbol):
    try:
        # Try to get currency from ticker info
        info = ticker_obj.info
        currency = info.get('currency', 'USD')
        
        # Check if it's an Indian stock (NSE or BSE)
        if '.NS' in stock_symbol or '.BO' in stock_symbol or currency == 'INR':
            return '₹', 'INR'
        elif currency == 'USD':
            return '$', 'USD'
        elif currency == 'EUR':
            return '€', 'EUR'
        elif currency == 'GBP':
            return '£', 'GBP'
        elif currency == 'JPY':
            return '¥', 'JPY'
        else:
            return '$', currency  # Default to $ but use the actual currency code
    except:
        # Default to USD if we can't determine
        return '$', 'USD'

# Function to get predictions from cloud API
def get_predictions(ticker, days=30):
    try:
        url = 'https://bullseye-price-predictor-281769648388.us-central1.run.app/predict'
        payload = {
            "ticker": ticker,
            "days": days
        }
        headers = {'Content-Type': 'application/json'}
        
        with st.spinner(f'Getting predictions for {ticker}...'):
            response = requests.post(url, json=payload, headers=headers)
            
        if response.status_code == 200:
            data = response.json()
            
            # Check if 'predictions' key exists in the response
            if 'predictions' in data:
                predictions_data = data['predictions']
                
                # Verify format of predictions data
                if isinstance(predictions_data, list) and len(predictions_data) > 0:
                    # Handle predictions in expected format with date and price keys
                    if 'date' in predictions_data[0] and 'price' in predictions_data[0]:
                        dates = [pred['date'] for pred in predictions_data]
                        prices = [pred['price'] for pred in predictions_data]
                    # Handle alternative format (array of arrays)
                    elif isinstance(predictions_data[0], list) and len(predictions_data[0]) >= 2:
                        dates = [pred[0] for pred in predictions_data]
                        prices = [pred[1] for pred in predictions_data]
                    else:
                        st.error("Unexpected prediction data format. Please check API response.")
                        st.json(data)
                        return None
                    
                    return {
                        'dates': dates,
                        'prices': prices
                    }
                else:
                    st.error("Empty predictions data received from API")
                    st.json(data)
                    return None
            else:
                # Alternative format check - direct array in response
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and 'date' in data[0] and 'price' in data[0]:
                        dates = [pred['date'] for pred in data]
                        prices = [pred['price'] for pred in data]
                        return {
                            'dates': dates,
                            'prices': prices
                        }
                
                st.error("Unexpected API response format")
                st.json(data)
                return None
        else:
            st.error(f"Error from prediction API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting predictions: {str(e)}")
        return None

# Main app layout
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_query = st.text_input(
            'Enter Company Name or Ticker',
            'Reliance',
            help='Enter the company name (e.g., Reliance) or stock symbol (e.g., RELIANCE.NS)'
        )
    
    with col2:
        prediction_days = st.number_input(
            'Prediction Days',
            min_value=5,
            max_value=90,
            value=30,
            help='Number of days to predict into the future'
        )
    
    stock = None
    if company_query:
        suggestions = search_company(company_query)
        
        if suggestions:
            options = [f"{s['name']} ({s['symbol']} - {s['exchange']})" for s in suggestions]
            selected_option = st.selectbox(
                'Select Company',
                options=options,
                index=0,
                help='Select the correct company from the list'
            )
            
            stock = selected_option.split('(')[1].split(')')[0].split(' - ')[0].strip()
        else:
            st.warning("No companies found. Please try a different search term.")

if stock:
    try:
        # Get stock data from Yahoo Finance
        ticker = yf.Ticker(stock)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            st.error(f"No data available for {stock}")
        else:
            info = ticker.info
            
            # Determine currency symbol and code
            currency_symbol, currency_code = get_currency_symbol(ticker, stock)
            
            # Display key metrics
            st.markdown('<h2 class="subtitle">Key Metrics</h2>', unsafe_allow_html=True)
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                current_price = info.get('currentPrice', df['Close'].iloc[-1])
                prev_close = info.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else None)
                if current_price and prev_close:
                    price_change = ((current_price - prev_close) / prev_close * 100)
                    st.metric("Current Price", f"{currency_symbol}{current_price:,.2f}", f"{price_change:.2f}%")
                else:
                    st.metric("Current Price", f"{currency_symbol}{df['Close'].iloc[-1]:,.2f}", None)
            
            with metrics_col2:
                market_cap = info.get('marketCap')
                if market_cap:
                    # Format market cap based on size
                    if market_cap >= 1e9:
                        market_cap_str = f"{currency_symbol}{(market_cap / 1e9):,.2f}B"
                    else:
                        market_cap_str = f"{currency_symbol}{(market_cap / 1e6):,.2f}M"
                    st.metric("Market Cap", market_cap_str, None)
                else:
                    st.metric("Market Cap", "N/A", None)
            
            with metrics_col3:
                pe_ratio = info.get('trailingPE')
                if pe_ratio:
                    st.metric("P/E Ratio", f"{pe_ratio:.2f}", None)
                else:
                    st.metric("P/E Ratio", "N/A", None)
            
            with metrics_col4:
                volume = info.get('volume', df['Volume'].iloc[-1])
                if volume:
                    st.metric("Volume", f"{volume:,}", None)
                else:
                    st.metric("Volume", "N/A", None)
            
            # Get predictions from API
            prediction_data = get_predictions(stock, prediction_days)
            
            if prediction_data and 'dates' in prediction_data and 'prices' in prediction_data:
                # Extract data
                dates = prediction_data['dates']
                predicted_prices = prediction_data['prices']
                
                if predicted_prices and dates:
                    # Historical and Prediction Chart
                    st.markdown('<h2 class="subtitle">Price Analysis & Predictions</h2>', unsafe_allow_html=True)
                    
                    # Create figure with secondary y-axis
                    fig = make_subplots(
                        rows=2, 
                        cols=1,
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.08,
                        subplot_titles=(f'Stock Price: Historical & Prediction ({currency_code})', 'Trading Volume')
                    )
                    
                    # Add historical price trace
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            name='Historical Price',
                            line=dict(color='royalblue', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add prediction trace
                    prediction_dates = pd.to_datetime(dates)
                    fig.add_trace(
                        go.Scatter(
                            x=prediction_dates,
                            y=predicted_prices,
                            name='Predicted Price',
                            line=dict(color='firebrick', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Add volume as bar chart on second row
                    fig.add_trace(
                        go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            name='Volume',
                            marker=dict(color='lightblue')
                        ),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=800,
                        hovermode='x unified',
                        template='plotly_white',
                        xaxis2_title='Date',
                        yaxis_title=f'Price ({currency_code})',
                        yaxis2_title='Volume',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Show plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction summary
                    st.markdown('<h2 class="subtitle">Prediction Summary</h2>', unsafe_allow_html=True)
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        last_price = df['Close'].iloc[-1]
                        st.metric("Current Price", f"{currency_symbol}{last_price:.2f}")
                    
                    with summary_col2:
                        final_prediction = predicted_prices[-1]
                        st.metric("Final Predicted Price", f"{currency_symbol}{final_prediction:.2f}")
                    
                    with summary_col3:
                        price_change = ((final_prediction - last_price) / last_price) * 100
                        st.metric("Predicted Change", f"{price_change:.2f}%", 
                                delta_color="normal" if price_change >= 0 else "inverse")
                    
                    # Display prediction data in a table with improved formatting
                    st.markdown('<h3 class="subtitle">Detailed Predictions</h3>', unsafe_allow_html=True)
                    
                    # Create a DataFrame with formatted data
                    prediction_df = pd.DataFrame({
                        'Date': pd.to_datetime(prediction_dates),
                        'Predicted Price': predicted_prices
                    })
                    
                    # Format the DataFrame for display
                    prediction_df['Date'] = prediction_df['Date'].dt.strftime('%Y-%m-%d')
                    prediction_df['Predicted Price'] = prediction_df['Predicted Price'].apply(
                        lambda x: f"{currency_symbol}{x:,.2f}"
                    )
                    
                    # Add a column for day of week to make the table more informative
                    prediction_df['Day of Week'] = pd.to_datetime(prediction_dates).day_name()
                    
                    # Reorder columns
                    prediction_df = prediction_df[['Date', 'Day of Week', 'Predicted Price']]
                    
                    # Display the formatted table
                    st.dataframe(
                        prediction_df,
                        use_container_width=True,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", format="%Y-%m-%d"),
                            "Day of Week": st.column_config.TextColumn("Day"),
                            "Predicted Price": st.column_config.TextColumn(f"Price ({currency_code})")
                        }
                    )
                    
                    # Add option to download predictions as CSV
                    csv = prediction_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name=f"{stock}_predictions.csv",
                        mime="text/csv",
                    )
                    
                    # Add disclaimer
                    st.info("""
                        **Disclaimer**: These predictions are based on historical data and machine learning models.
                        They should not be used as the sole basis for investment decisions. 
                        Past performance is not indicative of future results.
                    """)
                
                else:
                    st.error("No prediction data available from the API.")
            else:
                st.error("Failed to get predictions from the API.")
            
            # Company info section
            with st.expander("Company Information"):
                if info:
                    st.subheader(f"About {info.get('longName', stock)}")
                    
                    if info.get('longBusinessSummary'):
                        st.write(info['longBusinessSummary'])
                    
                    company_info_cols = st.columns(2)
                    
                    with company_info_cols[0]:
                        st.write("**Industry:**", info.get('industry', 'N/A'))
                        st.write("**Sector:**", info.get('sector', 'N/A'))
                        st.write("**Country:**", info.get('country', 'N/A'))
                        st.write("**Exchange:**", info.get('exchange', 'N/A'))
                        st.write("**Currency:**", info.get('currency', 'N/A'))
                    
                    with company_info_cols[1]:
                        st.write("**Website:**", info.get('website', 'N/A'))
                        st.write("**Employees:**", f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A')
                        st.write("**CEO:**", info.get('companyOfficers', [{}])[0].get('name', 'N/A') if info.get('companyOfficers') else 'N/A')
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)  # Show detailed error information for debugging
else:
    st.info("Please enter a company name or stock symbol to begin.")

# Footer
st.markdown("""
---
**Disclaimer**: This application is for educational purposes only and should not be considered financial advice.
Stock predictions are based on historical data and machine learning models, which may not accurately predict future performance.
""")
