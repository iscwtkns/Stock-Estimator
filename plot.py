import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "AAPL_hourly_data.csv"  # Replace with your file path
data = pd.read_csv(file_path)

# Ensure 'Date' is parsed as a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date (if not already sorted)
data.sort_values('Date', inplace=True)

# Plotting stock data
plt.figure(figsize=(12, 6))

# Plot 'Close' price over time
plt.plot(data['Date'], data['Close'], label='Close Price', color='blue', linewidth=2)
plt.plot(data['Date'], data['Open'], label='Open Price', color='red', linewidth=2)
plt.plot(data['Date'], data['Low'], label='Low Price', color='orange', linewidth=2)
plt.plot(data['Date'], data['High'], label='High Price', color='green', linewidth=2)
# Customize the plot
plt.title("Stock Price Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', fontsize=10)

# Display the plot
plt.show()
