import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

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

# Function to get predictions from cloud API
def get_predictions(ticker, days=180):
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

st.set_page_config(layout="wide", page_title="StoX - AI Stock Analysis")
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        padding: 2rem;
    }
    .stTitle {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1e3d59 !important;
        margin-bottom: 2rem !important;
    }
    .stSubheader {
        color: #17a2b8 !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<h1 class="stTitle">StoX</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.5rem; color: #666;">AI-based Stock Analysis & Prediction</p>', unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns([2,6,2])
    with col2:
        company_query = st.text_input(
            'Enter Company Name or Ticker',
            'Apple',
            help='Enter the company name (e.g., Apple) or stock symbol (e.g., AAPL)'
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
        ticker = yf.Ticker(stock)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = ticker.history(start=start_date, end=end_date) 
        if df.empty:
            st.error(f"No data available for {stock}")
        else:
            info = ticker.info
            # Display metrics only if I have valid data by the way I have valid data and Remember to Re-Train model
            if info:
                st.markdown('<h2 class="stSubheader">Key Metrics</h2>', unsafe_allow_html=True)
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)   
                with metrics_col1:
                    current_price = info.get('currentPrice')
                    prev_close = info.get('previousClose')
                    if current_price and prev_close:
                        price_change = ((current_price - prev_close) / prev_close * 100)
                        st.metric("Current Price", f"{current_price:,.2f}", f"{price_change:.2f}%")
                    else:
                        st.metric("Current Price", "N/A", "N/A")  
                with metrics_col2:
                    market_cap = info.get('marketCap')
                    if market_cap:
                        st.metric("Market Cap", f"{(market_cap / 1e9):,.2f}B", None)
                    else:
                        st.metric("Market Cap", "N/A", None)
                with metrics_col3:
                    pe_ratio = info.get('trailingPE')
                    if pe_ratio:
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}", None)
                    else:
                        st.metric("P/E Ratio", "N/A", None)
                with metrics_col4:
                    volume = info.get('volume')
                    if volume:
                        st.metric("Volume", f"{volume:,}", None)
                    else:
                        st.metric("Volume", "N/A", None)
                with st.expander("Detailed Stock Information"):
                    col1, col2 = st.columns(2) 
                    with col1:
                        st.write(f"Stock Name: {info.get('longName', stock)}")
                        st.write(f"Stock Code: {stock}")
                        st.write(f"Current Price: {ticker.info.get('currentPrice', 'N/A')} ")
                        st.write(f"Previous Close: {ticker.info.get('previousClose', 'N/A')} ")
                        st.write(f"Quote Change: {((ticker.info.get('currentPrice', 0) - ticker.info.get('previousClose', 0)) / ticker.info.get('previousClose', 1) * 100):.2f}%")
                        st.write(f"52-Week High: {ticker.info.get('fiftyTwoWeekHigh', 'N/A')} ")
                        st.write(f"52-Week Low: {ticker.info.get('fiftyTwoWeekLow', 'N/A')} ")
                        st.write(f"Open Price: {ticker.info.get('open', 'N/A')} ")
                        st.write(f"Day High: {ticker.info.get('dayHigh', 'N/A')} ")
                        st.write(f"Day Low: {ticker.info.get('dayLow', 'N/A')} ")
                    with col2:
                        st.write(f"Trading Volume: {ticker.info.get('volume', 'N/A'):,} shares")
                        st.write(f"Trading Value: {(ticker.info.get('volume', 0) * ticker.info.get('currentPrice', 0) / 1e9):.2f} billion ")
                        st.write(f"Market Cap: {(ticker.info.get('marketCap', 0) / 1e9):.2f} billion ")
                        st.write(f"Shares Outstanding: {(ticker.info.get('sharesOutstanding', 0) / 1e9):.2f} billion shares")
                        st.write(f"Float Shares: {(ticker.info.get('floatShares', 0) / 1e9):.2f} billion shares")
                        st.write(f"EPS (TTM): {ticker.info.get('trailingEps', 'N/A')}")
                        st.write(f"Forward EPS: {ticker.info.get('forwardEps', 'N/A')}")
                        st.write(f"P/E Ratio (TTM): {ticker.info.get('trailingPE', 'N/A'):.2f}")
                        st.write(f"Forward P/E: {ticker.info.get('forwardPE', 'N/A'):.2f}")
                        st.write(f"Price-to-Book Ratio: {ticker.info.get('priceToBook', 'N/A'):.2f}")
            # Technical Analysis Section
            st.markdown('<h2 class="stSubheader">Technical Analysis</h2>', unsafe_allow_html=True)
            time_periods = {
                "1 Month": 30,
                "6 Months": 180,
                "1 Year": 365,
                "5 Year": 1825,
            }
            selected_period = st.selectbox(
                "Select Time Period",
                options=list(time_periods.keys()),
                index=0
            )
            period_days = time_periods[selected_period]
            period_start = end_date - timedelta(days=period_days)
            df_period = ticker.history(start=period_start, end=end_date)
            if not df_period.empty:
                fig = make_subplots(
                    rows=3, cols=1,
                    row_heights=[0.3, 0.4, 0.3], 
                    vertical_spacing=0.05,
                    subplot_titles=(f'{selected_period} Mountain View', f'{selected_period} Price Analysis', 'Volume')
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_period.index,
                        y=df_period['Close'],
                        fill='tonexty',
                        name='Mountain View',
                        line=dict(color='rgb(0, 128, 96)'),
                        fillcolor='rgba(0, 128, 96, 0.3)'
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Candlestick(
                        x=df_period.index,
                        open=df_period['Open'],
                        high=df_period['High'],
                        low=df_period['Low'],
                        close=df_period['Close'],
                        name='OHLC'
                    ),
                    row=2, col=1
                )
                if period_days >= 100:
                    ma100 = df_period['Close'].rolling(window=100).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=df_period.index,
                            y=ma100,
                            name='100MA',
                            line=dict(color='red', width=1)
                        ),
                        row=2, col=1
                    )
                # volume bars
                fig.add_trace(
                    go.Bar(
                        x=df_period.index,
                        y=df_period['Volume'],
                        name='Volume',
                        marker=dict(color='rgba(0, 0, 255, 0.5)')
                    ),
                    row=3, col=1
                )
                fig.update_layout(
                    title=f'{stock} Stock Analysis - {selected_period}',
                    yaxis_title='Stock Price (USD)',
                    yaxis2_title='Stock Price (USD)',
                    yaxis3_title='Volume',
                    xaxis_rangeslider_visible=False,
                    template='plotly_white',
                    height=1000,  
                    showlegend=True
                )
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
                
                st.plotly_chart(fig, use_container_width=True)
                
            # AI Predictions Section
            if len(df) >= 100:  # Only show predictions if we have enough data
                st.markdown('<h2 class="stSubheader">AI Price Predictions</h2>', unsafe_allow_html=True)
                
                try:
                    # Get predictions from Cloud API
                      prediction_data = get_predictions(stock, 180)  # Request 180 days of predictions

                      if prediction_data and 'dates' in prediction_data and 'prices' in prediction_data:
                      # Extract prediction data
                        prediction_dates = pd.to_datetime(prediction_data['dates'])
                        predicted_prices = prediction_data['prices']
    
                   # Section 1: Current Predictions vs Actual
                     fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                     x=df.index[-180:],
        y=df['Close'].values[-180:],
        name='Actual Price',
        line=dict(color='blue', width=2)
    ))
    
    # Find matching date range in predictions if available
    matching_pred_indices = []
    for i, date in enumerate(prediction_dates):
        # Convert both timestamps to naive or convert both to aware
        # Option 1: Convert both to naive (removing timezone info)
        naive_date = date.replace(tzinfo=None)
        df_naive_index = df.index.tz_localize(None)
        
        if naive_date in df_naive_index:
            matching_pred_indices.append(i)
                                
                        if matching_pred_indices:
                            matching_dates = prediction_dates[matching_pred_indices]
                            matching_prices = [predicted_prices[i] for i in matching_pred_indices]
                            
                            fig_pred.add_trace(go.Scatter(
                                x=matching_dates,
                                y=matching_prices,
                                name='Predicted Price (Historical)',
                                line=dict(color='orange', width=2)
                            ))
                        
                        fig_pred.update_layout(
                            title='AI Price Predictions vs Actual Prices',
                            xaxis_title='Date',
                            yaxis_title='Stock Price',
                            template='plotly_white',
                            height=400
                        ) 
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Section 2: Future Predictions with Volatility
                        # Calculate historical volatility
                        historical_returns = np.log(df['Close'] / df['Close'].shift(1))
                        volatility = historical_returns.std() * np.sqrt(252)
                        
                        # Future prediction chart
                        fig_future = go.Figure()
                        fig_future.add_trace(go.Scatter(
                            x=df.index[-180:],
                            y=df['Close'].values[-180:],
                            name='Actual Price',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Get future dates from the prediction data
                        future_indices = [i for i, date in enumerate(prediction_dates) if date > df.index[-1]]
                        future_dates = prediction_dates[future_indices]
                        future_prices = [predicted_prices[i] for i in future_indices]
                        
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_prices,
                            name='Predicted Price (Next 6 Months)',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Calculate confidence intervals
                        upper_bound = [price * (1 + volatility) for price in future_prices]
                        lower_bound = [price * (1 - volatility) for price in future_prices]
                        
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=upper_bound,
                            fill=None,
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0)'),
                            showlegend=False
                        ))
                        
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=lower_bound,
                            fill='tonexty',
                            mode='lines',
                            line=dict(color='rgba(255,0,0,0)'),
                            name='Confidence Interval',
                            fillcolor='rgba(255,0,0,0.1)'
                        ))
                        
                        fig_future.update_layout(
                            title='AI Price Predictions with Confidence Intervals',
                            xaxis_title='Date',
                            yaxis_title='Stock Price (USD)',
                            template='plotly_white',
                            height=600,
                            hovermode='x unified',
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        st.plotly_chart(fig_future, use_container_width=True)
                        
                        # Add prediction insights
                        current_price = df['Close'].iloc[-1]
                        final_pred_price = future_prices[-1] if future_prices else 0
                        price_change = ((final_pred_price - current_price) / current_price) * 100
                        
                        st.write("### Prediction Insights")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"{current_price:.2f}")
                        with col2:
                            st.metric("Predicted Price (6m)", f"{final_pred_price:.2f}")
                        with col3:
                            st.metric("Expected Change", f"{price_change:.1f}%") 
                        st.info("Note: Predictions include confidence intervals based on historical volatility. The shaded area represents the potential price range with 68% confidence.")
                    else:
                        st.warning("Failed to get predictions from the API. Using historical data only.")
                except Exception as e:
                    st.error(f"Error processing data for predictions: {str(e)}")
                    
            # PDF Export Function
            def create_stock_report_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    spaceAfter=30
                )
                elements.append(Paragraph(f"Stock Analysis Report - {stock}", title_style))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph("Key Metrics", styles['Heading2']))
                metrics_data = [
                    ["Metric", "Value"],
                    ["Stock Name", info.get('longName', stock)],
                    ["Stock Code", stock],
                    ["Current Price", f"{ticker.info.get('currentPrice', 'N/A')} "],
                    ["Previous Close", f"{ticker.info.get('previousClose', 'N/A')} "],
                    ["Quote Change", f"{((ticker.info.get('currentPrice', 0) - ticker.info.get('previousClose', 0)) / ticker.info.get('previousClose', 1) * 100):.2f}%"],
                    ["52-Week High", f"{ticker.info.get('fiftyTwoWeekHigh', 'N/A')} "],
                    ["52-Week Low", f"{ticker.info.get('fiftyTwoWeekLow', 'N/A')} "],
                    ["Open Price", f"{ticker.info.get('open', 'N/A')} USD"],
                    ["Day High", f"{ticker.info.get('dayHigh', 'N/A')} USD"],
                    ["Day Low", f"{ticker.info.get('dayLow', 'N/A')} USD"],
                    ["Trading Volume", f"{ticker.info.get('volume', 'N/A'):,} shares"],
                    ["Trading Value", f"{(ticker.info.get('volume', 0) * ticker.info.get('currentPrice', 0) / 1e9):.2f} billion "],
                    ["Market Cap", f"{(ticker.info.get('marketCap', 0) / 1e9):.2f} billion "],
                    ["Shares Outstanding", f"{(ticker.info.get('sharesOutstanding', 0) / 1e9):.2f} billion shares"],
                    ["Float Shares", f"{(ticker.info.get('floatShares', 0) / 1e9):.2f} billion shares"],
                    ["EPS (TTM)", f"{ticker.info.get('trailingEps', 'N/A')}"],
                    ["Forward EPS", f"{ticker.info.get('forwardEps', 'N/A')}"],
                    ["P/E Ratio (TTM)", f"{ticker.info.get('trailingPE', 'N/A'):.2f}"],
                    ["Forward P/E", f"{ticker.info.get('forwardPE', 'N/A'):.2f}"],
                    ["Price-to-Book Ratio", f"{ticker.info.get('priceToBook', 'N/A'):.2f}"]
                ]
                table = Table(metrics_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 20))
                # Here I add AI Predictions section to the PDF if available
                if 'final_pred_price' in locals():
                    elements.append(Paragraph("AI Predictions", styles['Heading2']))
                    predictions_data = [
                        ["Metric", "Value"],
                        ["Current Price", f"{current_price:.2f}"],
                        ["Predicted Price (6m)", f"{final_pred_price:.2f}"],
                        ["Expected Change", f"{price_change:.1f}%"],
                        ["Historical Volatility", f"{volatility*100:.1f}%"]
                    ]
                    pred_table = Table(predictions_data)
                    pred_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, -1), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(pred_table)
                # Add report generation timestamp
                elements.append(Spacer(1, 30))
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elements.append(Paragraph(f"Report generated on: {timestamp}", styles['Normal']))
                doc.build(elements)
                buffer.seek(0)
                return buffer
            # Add download button
            st.markdown('<h2 class="stSubheader">Export Report</h2>', unsafe_allow_html=True)
            pdf_buffer = create_stock_report_pdf()
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"{stock}_analysis_report.pdf",
                mime="application/pdf",
                key='download_button',
                help="Download a detailed PDF report of the stock analysis"
            )
    except Exception as e:
        st.error(f"Error processing data for {stock}: {str(e)}")
else:
    st.info("Please enter a stock symbol to begin analysis.")
# Detailed Information Section of Company
st.title("📈 Detailed Information of Company")
navigation = st.tabs(["Stock News", "About", "Contact"])
with navigation[0]:
    st.subheader("Search Stock News")
    if stock and info:
        company_name = info.get('longName', stock)
        query = company_name  
    else:
        query = "Apple"  
    query = st.text_input("Enter stock/company name:", value=query)
    API_KEY = "84f6a80ac55c464c930ce84304485380"  
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    def fetch_stock_news(query, api_key):
        """Fetch stock-related news from NewsAPI."""
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "apiKey": api_key,
            "language": "en",
            "pageSize": 10,
        }
        response = requests.get(NEWS_API_URL, params=params)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"Error fetching news: {response.status_code}")
            return []
    if st.button("Search News"):
        st.subheader(f"Latest News for '{query}'")
        articles = fetch_stock_news(query, API_KEY)
        
        if articles:
            for article in articles:
                st.write("### " + article["title"])
                st.write(f"Source: {article['source']['name']}")
                st.write(f"Published: {article['publishedAt']}")
                st.write(article["description"])
                st.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)
                st.write("---")
        else:
            st.warning(f"No news found for '{query}'.")

        

# About Section
with navigation[1]:
    if stock and info:
        st.subheader(f"About {info.get('longName', stock)}")   
        # Company Description
        if info.get('longBusinessSummary'):
            st.write("### Business Summary")
            st.write(info['longBusinessSummary'])
        # Key Company Information
        st.write("### Company Details")
        company_details = {
            "Industry": info.get('industry', 'N/A'),
            "Sector": info.get('sector', 'N/A'),
            "Full Time Employees": f"{info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else 'N/A',
            "Country": info.get('country', 'N/A'),
            "State": info.get('state', 'N/A'),
            "City": info.get('city', 'N/A'),
        } 
        # Company details in two columns
        col1, col2 = st.columns(2)
        for i, (key, value) in enumerate(company_details.items()):
            if i % 2 == 0:
                col1.write(f"*{key}:* {value}")
            else:
                col2.write(f"*{key}:* {value}")
        # Financial Information
        st.write("### Financial Overview")
        financial_metrics = {
            "Revenue Growth": f"{info.get('revenueGrowth', 'N/A')*100:.2f}%" if info.get('revenueGrowth') else 'N/A',
            "Gross Margins": f"{info.get('grossMargins', 'N/A')*100:.2f}%" if info.get('grossMargins') else 'N/A',
            "Operating Margins": f"{info.get('operatingMargins', 'N/A')*100:.2f}%" if info.get('operatingMargins') else 'N/A',
            "Profit Margins": f"{info.get('profitMargins', 'N/A')*100:.2f}%" if info.get('profitMargins') else 'N/A',
        }
        # Financial metrics in two columns
        col1, col2 = st.columns(2)
        for i, (key, value) in enumerate(financial_metrics.items()):
            if i % 2 == 0:
                col1.write(f"*{key}:* {value}")
            else:
                col2.write(f"*{key}:* {value}")
    else:
        st.info("Please select a company to view detailed information.")
# Contact Section
with navigation[2]:
    if stock and info:
        st.subheader(f"Contact Information for {info.get('longName', stock)}")
        contact_info = {
            "Website": info.get('website', 'N/A'),
            "Phone": info.get('phone', 'N/A'),
            "Address": f"{info.get('address1', '')}, {info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}"
        }
        for key, value in contact_info.items():
            if value and value != 'N/A':
                if key == "Website" and value != 'N/A':
                    st.markdown(f"*{key}:* [{value}]({value})")
                else:
                    st.write(f"*{key}:* {value}")
    else:
        st.info("Please select a company to view contact information.")
# Footer
st.markdown(
    """
    ---
    Disclaimer: The information provided on this website is for informational purposes only and should not be considered as financial advice.
    """
)
