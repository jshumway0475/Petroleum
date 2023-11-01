import streamlit as st
import pandas as pd
import yfinance as yf
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
import plotly.graph_objects as go

# Create a calendar instance
today = dt.date.today()
end_date = today + dt.timedelta(days=10*365.25)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2000-01-01', end=end_date).date

# Dictionary for month codes
month_codes = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J',
    5: 'K', 6: 'M', 7: 'N', 8: 'Q',
    9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# Function to generate the ticker for the oil futures price
def generate_oil_gas_ticker(month, year, oil_gas='oil'):
    '''
    Function to generate the ticker for oil or natural gas futures prices
    month: int, month of the futures contract
    year: int, year of the futures contract
    oil_gas: str, 'oil' or 'gas'
    '''    
    base_symbols = {'oil': 'CL', 'gas': 'NG'}
    base_symbol = base_symbols.get(oil_gas, 'CL')  # default to 'CL' if invalid oil_gas value is passed
    
    return f'{base_symbol}{month_codes[month]}{str(year)[-2:]}.NYM'

# Function to generate a list of tickers for oil and natural gas futures prices
def oil_gas_ticker_dict(date: str, years: int=5):
    '''
    Function to generate dictionaries of tickers for oil and natural gas futures prices
    date: str, date in the format of 'YYYY-MM-DD'
    years: int, number of years of futures prices to include
    '''
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

# Function to calculate the expiry date for oil futures contracts
def get_expiry_date_for_oil(ticker):
    month_code = ticker[2]
    year = int('20' + ticker[3:5])
    month = [k for k, v in month_codes.items() if v == month_code][0] - 1
    
    expiration_date = dt.date(year, month, 25)
    
    # Subtract weekends and holidays
    while expiration_date.weekday() >= 5 or expiration_date in holidays:
        expiration_date -= dt.timedelta(days=1)
        
    # Subtract 3 days for the business rule
    for _ in range(3):
        expiration_date -= dt.timedelta(days=1)
        while expiration_date.weekday() >= 5 or expiration_date in holidays:
            expiration_date -= dt.timedelta(days=1)
    return expiration_date

# Function to calculate the expiry date for natural gas futures contracts
def get_expiry_date_for_gas(ticker):
    month_code = ticker[2]
    year = int('20' + ticker[3:5])
    month = [k for k, v in month_codes.items() if v == month_code][0]
    
    expiration_date = dt.date(year, month, 1)  # first day of the delivery month
    
    # Subtract 3 days for the business rule
    for _ in range(3):
        expiration_date -= dt.timedelta(days=1)
        while expiration_date.weekday() >= 5 or expiration_date in holidays:
            expiration_date -= dt.timedelta(days=1)
    return expiration_date

# Function to fetch data from Yahoo Finance
def fetch_data(ticker, specific_date):
    start_date = specific_date - dt.timedelta(days=1) # a day before to ensure the specific_date is included
    end_date = specific_date + dt.timedelta(days=1)   # a day after to ensure the specific_date is included
    data = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    
    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError(f'Data not available for ticker {ticker} on {specific_date}')
    return data

# Function to extract the close price and volume from the data
def extract_data(data, specific_date):
    specific_date_str = specific_date.strftime('%Y-%m-%d')
    if specific_date_str in data.index:
        entry = data.loc[specific_date_str]
        return entry['Close'], entry['Volume']
    else:
        return None, None

# Function to combine the data into a dataframe
def get_combined_data(input_date: str, years: int=5):
    specific_date = dt.datetime.strptime(input_date, '%Y-%m-%d')
    oil_dict, gas_dict = oil_gas_ticker_dict(input_date, years)
    data_list = []

    for date, oil_ticker in oil_dict.items():
        gas_ticker = gas_dict.get(date)
        try:
            oil_data = fetch_data(oil_ticker, specific_date)
            gas_data = fetch_data(gas_ticker, specific_date)
            oil_close, oil_volume = extract_data(oil_data, specific_date)
            gas_close, gas_volume = extract_data(gas_data, specific_date)
        except ValueError:
            try:
                oil_expiry_date = get_expiry_date_for_oil(oil_ticker)
                gas_expiry_date = get_expiry_date_for_gas(gas_ticker)
                oil_data = fetch_data('CL=F', oil_expiry_date)
                gas_data = fetch_data('NG=F', gas_expiry_date)
                oil_close, oil_volume = extract_data(oil_data, oil_expiry_date)
                gas_close, gas_volume = extract_data(gas_data, gas_expiry_date)
            except ValueError:
                oil_close, oil_volume = None, None
                gas_close, gas_volume = None, None
        
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

def plot_prices_with_tooltips(dataframe):
    # Create a trace for Oil prices
    trace_oil = go.Scatter(
        x=dataframe['Date'],
        y=dataframe['Oil Close Price'],
        mode='lines',
        name='Oil Price',
        line=dict(color='green', width=4.0),
        yaxis='y1',
        hovertemplate="Date: %{x}<br>Oil Price: $%{y:.2f}<extra></extra>"
    )
    
    # Create a trace for Gas prices
    trace_gas = go.Scatter(
        x=dataframe['Date'],
        y=dataframe['Gas Close Price'],
        mode='lines',
        name='Gas Price',
        line=dict(color='red', width=4.0),
        yaxis='y2',
        hovertemplate="Date: %{x}<br>Gas Price: $%{y:.2f}<extra></extra>"
    )
    
    layout = go.Layout(
        showlegend=False,  # Turn off the legend
        yaxis=dict(
            title='Oil Price', 
            color='green',
            tickprefix='$',
            tickformat='.2f',
            showgrid=True,     # Gridlines for y-axis
            zeroline=True,     # Show the 0-line
            gridcolor='lightgray', # Gridline color
            linecolor='black', # Axis line color
            mirror=True        # Mirror axis line
        ),
        yaxis2=dict(
            title='Gas Price', 
            color='red',
            overlaying='y',
            side='right',
            tickprefix='$',
            tickformat='.2f',
            showgrid=False,
            zeroline=True,     # Show the 0-line
            gridcolor='lightgray', # Gridline color
            linecolor='black', # Axis line color
            mirror='ticks'     # Mirror axis line with ticks
        ),
        xaxis=dict(
            showgrid=True,     # Gridlines for x-axis
            zeroline=True,     # Show the 0-line
            gridcolor='lightgray', # Gridline color
            linecolor='black', # Axis line color
            mirror=True        # Mirror axis line
        ),
        plot_bgcolor='white'
    )
    
    fig = go.Figure(data=[trace_oil, trace_gas], layout=layout)
    st.plotly_chart(fig)

# Streamlit UI
st.write('This app fetches the oil and natural gas futures prices from Yahoo Finance')
st.write('If a contract is expired, it will fetch the settlement on the final day of trading')
st.markdown('[Oil Contract Expiry](https://www.eia.gov/dnav/pet/TblDefs/pet_pri_fut_tbldef2.asp)')
st.markdown('[Nat Gas Contract Expiry](https://www.eia.gov/dnav/ng/TblDefs/ng_pri_fut_tbldef2.asp)')
yesterday = today - dt.timedelta(days=1)
strip_date = st.date_input('Strip Date', yesterday, min_value=yesterday - dt.timedelta(days=365), max_value=yesterday)
years = st.slider('Years', 1, 10, 5)

if st.button('Get Data'):
    with st.spinner('Fetching data...'):
        result = get_combined_data(strip_date.strftime('%Y-%m-%d'), years)
    st.write(result)
    plot_prices_with_tooltips(result)

st.write('Adjust the inputs and click "Get Data" to fetch the data.')
