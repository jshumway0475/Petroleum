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

def fetch_data(ticker, specific_date):
    start_date = specific_date - dt.timedelta(days=1) # a day before to ensure the specific_date is included
    end_date = specific_date + dt.timedelta(days=1)   # a day after to ensure the specific_date is included
    data = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data

def extract_data(data, specific_date):
    specific_date_str = specific_date.strftime('%Y-%m-%d')
    if specific_date_str in data.index:
        entry = data.loc[specific_date_str]
        return entry['Close'], entry['Volume']
    else:
        return None, None

def get_combined_data(input_date: str, years: int=5):
    specific_date = dt.datetime.strptime(input_date, '%Y-%m-%d')
    oil_dict, gas_dict = oil_gas_ticker_dict(input_date, years)
    data_list = []

    for date, oil_ticker in oil_dict.items():
        gas_ticker = gas_dict.get(date)
        oil_data = fetch_data(oil_ticker, specific_date)
        gas_data = fetch_data(gas_ticker, specific_date)
        
        oil_close, oil_volume = extract_data(oil_data, specific_date)
        gas_close, gas_volume = extract_data(gas_data, specific_date)
        
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

today = dt.datetime.now().date()
yesterday = today - dt.timedelta(days=1)
strip_date = st.date_input('Strip Date', yesterday, min_value=yesterday - dt.timedelta(days=365), max_value=yesterday)
years = st.slider('Years', 1, 10, 5)

if st.button('Get Data'):
    result = get_combined_data(strip_date.strftime('%Y-%m-%d'), years)
    st.write(result)

st.write('Adjust the inputs and click "Get Data" to fetch the data.')
