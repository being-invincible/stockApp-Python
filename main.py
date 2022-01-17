import streamlit as st
from datetime import date
import pandas as pd

from plotly import graph_objs as go

# Importing fbprophet or similar time-series forecasting
import yfinance as yf
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams

# Specifying start date & today's date
start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

# Start Creating the web app with streamlit
# Title
st.title("Stock Web App")
st.write("A Streamlit stock trend prediction app for Educational and R&D purposes.")

# Hide
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style,unsafe_allow_html=True)

# Save the input Stock Ticker in type box
user_ticker = st.text_input('Enter Ticker :', 'AAPL')

# Defining a data loader function to download the selected stock ticker
# To Store the info so that data needs no reloading
@st.cache
def load_data(ticker):
    data = yf.download(user_ticker, start, )
    # To put date into the first column
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data = load_data(user_ticker)
data_load_state.text("Loading Data... Done!")

# Looking into the data frame via streamlit
st.write('Data from %s - %s' %(start, today))
if st.checkbox('Show dataframe'):
    st.write(data.tail())

# Function for ploting raw stock data
# Function for candle stick plot
def candle_plot():
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],open=data['Open'], high=data['High'],low=data['Low'], close=data['Close'])])
    fig.layout.update(title_text= "Raw Stock Plot",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=2,
                        label="2m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
            ])
        )),
        xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

candle_plot()

# Simple Moving Average
moving_avg_df = data.filter(['Date','Close'])
moving_avg_df['Moving Avg 100'] = moving_avg_df['Close'].rolling(100).mean()
moving_avg_df['Moving Avg 200'] = moving_avg_df['Close'].rolling(200).mean()

# Plot the moving avgerages
def moving_avg_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moving_avg_df['Date'], y=moving_avg_df['Close'], name='Stock Close'))
    fig.add_trace(go.Scatter(x=moving_avg_df['Date'], y=moving_avg_df['Moving Avg 100'], name='Moving Avg. 100 Days'))
    fig.add_trace(go.Scatter(x=moving_avg_df['Date'], y=moving_avg_df['Moving Avg 200'], name='Moving Avg. 200 Days'))
    fig.layout.update(title_text="Moving Averages Vs Close Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

st.subheader('Simple Moving Average Analysis')
moving_avg_plot()

# Slider for prediction day-range
n_days = st.slider("Days of Prediction:", 50, 100)
#period = n_months * 30

def model_call():
    # convert to TimeSeriesData object
    train_df = data.filter(['Date','Close'])
    train_df = train_df.rename({'Date': 'time', 'Close': 'value'}, axis='columns')

    time_series = TimeSeriesData(train_df)

    # import the param and model classes for STL model
    from kats.models.stlf import STLFModel, STLFParams

    # create a model param instance
    params = STLFParams(method='prophet', m=12)

    # create a prophet model instance
    m = STLFModel(time_series, params)

    # fit model simply by calling m.fit()
    m.fit()

    # make prediction for next 30 month
    forecast = m.predict(steps=n_days, include_history=False)

    forecast_plot(train_df, forecast)

    return forecast.tail()

def forecast_plot(train_df, forecast):
    plot_df = train_df.append(forecast)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df['time'], y=plot_df['value'], name='Actual Stock Closing', line ={'color':'deepskyblue'}))
    fig.add_trace(go.Scatter(x=plot_df['time'], y=plot_df['fcst'], name='Predicted Stock Closing', line ={'color':'limegreen'}))

    fig.layout.update(title_text="Forecast",
                  xaxis=dict(
                    rangeselector=dict(
                    buttons=list([
                    dict(count=1,
                        label="YTD",
                        step="year",
                        stepmode="todate"),
                    dict(step="all")
                        ])
                    )), xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

if st.button('Predict'):
    result = model_call()
    st.subheader("STLF Results -")
    st.write(result.head())