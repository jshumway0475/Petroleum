import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt

# Function to generate the ticker for the oil futures price
def generate_oil_gas_ticker(month, year, oil_gas='oil'):
    month_codes = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J',
        5: 'K', 6: 'M', 7: 'N', 8: 'Q',
        9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    base_symbols = {'oil': 'CL', 'gas': 'NG'}
    base_symbol = base_symbols.get(oil_gas, 'CL')
    return f'{base_symbol}{month_codes[month]}{str(year)[-2:]}.NYM'

def oil_gas_ticker_dict(date: str, years: int=5):
    date_dt = dt.datetime.strptime(date, '%Y-%m-%d')
    date_year = date_dt.year
    date_month = date_dt.month
    oil_tickers = {
        f"{year}-{month:02}": generate_oil_gas_ticker(month, year, 'oil') 
        for year in range(date_year, date_year + years) 
        for month in range(1, 13) 
        if not (year == date_year and month < date_month)
    }
    gas_tickers = {
        f"{year}-{month:02}": generate_oil_gas_ticker(month, year, 'gas') 
        for year in range(date_year, date_year + years) 
        for month in range(1, 13) 
        if not (year == date_year and month < date_month)
    }
    return oil_tickers, gas_tickers

def fetch_data(ticker):
    data = yf.Ticker(ticker).history(period="1y")
    return data

def extract_data(data):
    if not data.empty:
        return data.iloc[-1]['Close'], data.iloc[-1]['Volume']
    else:
        return None, None

def get_combined_data(date: str, years: int=5):
    oil_dict, gas_dict = oil_gas_ticker_dict(date, years)
    data_list = []
    for date, oil_ticker in oil_dict.items():
        gas_ticker = gas_dict.get(date)
        oil_data = fetch_data(oil_ticker)
        gas_data = fetch_data(gas_ticker)
        oil_close, oil_volume = extract_data(oil_data)
        gas_close, gas_volume = extract_data(gas_data)
        data_list.append({
            'Date': date,
            'Oil Ticker': oil_ticker,
            'Gas Ticker': gas_ticker,
            'Oil Close Price': oil_close,
            'Gas Close Price': gas_close,
            'Oil Volume': oil_volume,
            'Gas Volume': gas_volume
        })
    return pd.DataFrame(data_list)

# Streamlit UI
st.title("Oil & Gas Futures Prices")

strip_date = st.date_input('Strip Date', '2023-10-30')
years = st.slider('Years', 1, 10, 5)

if st.button('Get Data'):
    result = get_combined_data(strip_date.strftime('%Y-%m-%d'), years)
    st.write(result)

st.write('Adjust the inputs and click "Get Data" to fetch the data.')