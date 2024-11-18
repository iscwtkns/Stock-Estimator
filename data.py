from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Step 1: Initialize Alpha Vantage API
API_KEY = "ES1HU2TQM0SATU23"  # Replace with your Alpha Vantage API key
symbol = "AAPL"          # Stock symbol
interval = "60min"       # Hourly data
output_size = "full"     # Get full year of data

# Step 2: Fetch data
def fetch_hourly_data(symbol, interval, output_size, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=output_size)
    return data

# Main function
if __name__ == "__main__":
    stock_data = fetch_hourly_data(symbol, interval, output_size, API_KEY)
    stock_data = stock_data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data.sort_index(inplace=True)

    # Save to CSV for future use
    stock_data.to_csv(f"{symbol}_hourly_data.csv")

    print(stock_data.head())
