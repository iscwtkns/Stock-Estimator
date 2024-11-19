#import yfinance as yf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler

symbols = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp.
    "GOOGL", # Alphabet Inc. (Google)
    "AMZN",  # Amazon.com, Inc.
    "TSLA",  # Tesla Inc.
    "META",  # Meta Platforms, Inc. (formerly Facebook)
    "NVDA",  # NVIDIA Corporation
    "AMD",   # Advanced Micro Devices, Inc.
    "NFLX",  # Netflix, Inc.
    "INTC",  # Intel Corporation
    "BA",    # The Boeing Company
    "GE",    # General Electric Company
    "IBM",   # International Business Machines Corporation
    "DIS",   # The Walt Disney Company
    "JNJ",   # Johnson & Johnson
    "PG",    # Procter & Gamble Co.
    "XOM",   # Exxon Mobil Corporation
    "CVX",   # Chevron Corporation
    "WMT",   # Walmart Inc.
    "KO",    # Coca-Cola Company
    "PEP",   # PepsiCo, Inc.
    "MCD",   # McDonald's Corporation
    "T",     # AT&T Inc.
    "VZ",    # Verizon Communications Inc.
    "SPY",   # SPDR S&P 500 ETF Trust (represents S&P 500 index)
    "IWM",   # iShares Russell 2000 ETF
    "QQQ",   # Invesco QQQ Trust (tracks the NASDAQ 100)
    "GLD",   # SPDR Gold Shares ETF
    "SLV",   # iShares Silver Trust
    "BTC-USD", # Bitcoin USD
    "ETH-USD", # Ethereum USD
    "SPCE",  # Virgin Galactic Holdings, Inc.
    "PYPL",  # PayPal Holdings, Inc.
    "SQ",    # Block, Inc. (formerly Square)
    "CRM",   # Salesforce, Inc.
    "ADBE",  # Adobe Inc.
    "CSCO",  # Cisco Systems, Inc.
    "INTU",  # Intuit Inc.
    "GILD",  # Gilead Sciences, Inc.
    "AMAT",  # Applied Materials, Inc.
    "TSM",   # Taiwan Semiconductor Manufacturing Company
    "NKE",   # NIKE, Inc.
    "LMT",   # Lockheed Martin Corporation
    "MMM",   # 3M Company
    "CAT",   # Caterpillar Inc.
    "CVS",   # CVS Health Corporation
    "HSBC",  # HSBC Holdings plc
    "MS",    # Morgan Stanley
    "GS",    # The Goldman Sachs Group, Inc.
    "JPM",   # JPMorgan Chase & Co.
    "WFC",   # Wells Fargo & Co.
    "C",     # Citigroup Inc.
    "BMY",   # Bristol-Myers Squibb Company
    "MRK",   # Merck & Co., Inc.
    "PFE",   # Pfizer Inc.
    "AMGN",  # Amgen Inc.
    "ABBV",  # AbbVie Inc.
    "LLY",   # Eli Lilly and Co.
    "UNH",   # UnitedHealth Group Incorporated
    "CVX",   # Chevron Corporation
    "COP",   # ConocoPhillips
    "MRO",   # Marathon Oil Corporation
    "EOG",   # EOG Resources, Inc.
    "HAL",   # Halliburton Company
    "OXY",   # Occidental Petroleum Corporation
    "BP",    # BP plc
    "ENB",   # Enbridge Inc.
    "TSLA",  # Tesla Inc.
    "NIO",   # NIO Inc.
    "BYDDF", # BYD Company Limited
    "XPEV",  # Xpeng Inc.
    "LI",    # Li Auto Inc.
    "RIVN",  # Rivian Automotive, Inc.
    "LCID",  # Lucid Group, Inc.
    "F",     # Ford Motor Company
    "GM",    # General Motors Company
    "SPY",   # SPDR S&P 500 ETF Trust
    "LULU",  # Lululemon Athletica Inc.
    "ZUMZ",  # Zumiez Inc.
    "NKE",   # NIKE, Inc.
    "BABA",  # Alibaba Group Holding Ltd
    "JD",    # JD.com, Inc.
    "TCEHY", # Tencent Holdings Limited
    "SAP",   # SAP SE
    "WBA",   # Walgreens Boots Alliance, Inc.
    "V",     # Visa Inc.
    "MA",    # Mastercard Incorporated
    "AXP",   # American Express Company
    "PYPL",  # PayPal Holdings, Inc.
    "SQ",    # Block, Inc.
    "NDAQ",  # Nasdaq, Inc.
    "WBA",   # Walgreens Boots Alliance, Inc.
    "ZTS",   # Zoetis Inc.
    "SYK",   # Stryker Corporation
    "MDT",   # Medtronic plc
    "ABT",   # Abbott Laboratories
    "ED",    # Consolidated Edison, Inc.
    "SO",    # Southern Company
    "DTE",   # DTE Energy Company
    "DUK",   # Duke Energy Corporation
    "AEP",   # American Electric Power
    "WEC",   # WEC Energy Group, Inc.
    "XEL",   # Xcel Energy Inc.
    "EXC",   # Exelon Corporation
    "NEE",   # NextEra Energy, Inc.
    "ES",    # Eversource Energy
    "SRE",   # Sempra Energy
    "PPL",   # PPL Corporation
    "PCG",   # PG&E Corporation
    "DUK",   # Duke Energy Corporation
    "AMT",   # American Tower Corporation
    "CCI",   # Crown Castle International Corp
    "SBAC",  # SBA Communications Corporation
    "PSA",   # Public Storage
    "SPG",   # Simon Property Group, Inc.
    "DLR",   # Digital Realty Trust, Inc.
    "O",     # Realty Income Corporation
    "REG",   # Regency Centers Corporation
    "ARE",   # Alexandria Real Estate Equities, Inc.
    "EQR",   # Equity Residential
    "ESS",   # Essex Property Trust, Inc.
    "VNO",   # Vornado Realty Trust
    "AVB",   # AvalonBay Communities, Inc.
    "BXP",   # Boston Properties, Inc.
    "JLL",   # Jones Lang Lasalle Incorporated
    "CBRE",  # CBRE Group, Inc.
    "HST",   # Host Hotels & Resorts, Inc.
    "KIM",   # Kimco Realty Corporation
    "CPT",   # Camden Property Trust
    "FRT",   # Federal Realty Investment Trust
    "AMT",   # American Tower Corporation
    "OHI",   # Omega Healthcare Investors, Inc.
    "SUI",   # Sun Communities, Inc.
    "FLO",   # Flowers Foods, Inc.
    "CTAS",  # Cintas Corporation
    "ROL",   # Rollins, Inc.
    "SYY",   # Sysco Corporation
    "LUV",   # Southwest Airlines Co.
    "DAL",   # Delta Air Lines, Inc.
    "UAL",   # United Airlines Holdings, Inc.
    "AAL",   # American Airlines Group, Inc.
    "ALK",   # Alaska Air Group, Inc.
    "JBLU",  # JetBlue Airways Corporation
    "SAVE",  # Spirit Airlines, Inc.
    "HA",    # Hawaiian Holdings, Inc.
    "SKYW",  # SkyWest, Inc.
    "LUV",   # Southwest Airlines Co.
    "TAP",   # Molson Coors Beverage Company
    "KO",    # Coca-Cola Company
    "PEP",   # PepsiCo, Inc.
    "SBUX",  # Starbucks Corporation
    "MCD",   # McDonald's Corporation
    "YUM",   # Yum! Brands, Inc.
    "CMG",   # Chipotle Mexican Grill, Inc.
    "QSR",   # Restaurant Brands International Inc.
    "WEN",   # The Wendy's Company
    "JACK",  # Jack in the Box Inc.
]

# Process data
'''
def processYearlyStockData(symbol):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = yf.download(symbol, period="1y", interval="1d")
    scaled_data = scaler.fit_transform(data[['Close']])
    yearlyCloseValues = scaled_data.astype(float)
    xvalues = [yearlyCloseValues[0][0]]
    for value in yearlyCloseValues[1:]:
        if len(xvalues) < yearlyCloseValues.size-2:
            xvalues.append(value[0])
    yvalues = [yearlyCloseValues[yearlyCloseValues.size-1][0]]
    return (xvalues, yvalues)

def CompileData(symbols):
    trainingPercentage = 0.7
    validationPercentage = 0.2
    trainX = []
    trainY = []
    validationX = []
    validationY = []
    testX = []
    testY = []
    for symbol in symbols:
        xvalues, yvalues = processYearlyStockData(symbol)
        if len(trainX) < trainingPercentage*len(symbols):
            trainX.append(xvalues)
            trainY.append(yvalues)
        elif len(validationX) < validationPercentage*len(symbols):
            validationX.append(xvalues)
            validationY.append(yvalues)
        else:
            testX.append(xvalues)
            testY.append(yvalues)
    
    training = [trainX, trainY]
    validation = [validationX, validationY]
    test = [testX, testY]

    return (training,validation,test)

def save_processed_data(xvalues, yvalues, filename):
    # Combine inputs and outputs into a DataFrame
    data = pd.DataFrame({
        "Inputs": xvalues,
        "Targets": yvalues
    })
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
training, validation, test = CompileData(symbols)
'''
def load_processed_data(file_name):
    data = pd.read_csv(file_name)
    
    # Parse the inputs column into a list of arrays
    xvalues = [np.fromstring(row.strip("[]"), sep=",") for row in data['Inputs']]
    
    # Determine the target length (e.g., length of the shortest row)
    min_length = 200
    
    # Truncate rows to the desired length
    xvalues = np.array([row[-min_length:] for row in xvalues])
    
    # Parse the targets column
    yvalues = np.array([np.fromstring(str(row).strip("[]"), sep=",") for row in data['Targets']])
    
    return xvalues, yvalues

training, validation = load_processed_data('trainingData'), load_processed_data('validationData')
x_train = training[0]
y_train = training[1]
x_val = validation[0]
y_val = validation[1]
x_train = np.array(x_train).reshape((len(x_train), 200, 1))  # 60 timesteps, 1 feature
x_val = np.array(x_val).reshape((len(x_val), 200, 1))
y_train = np.array(y_train).reshape((len(y_train), 1))
y_val = np.array(y_val).reshape((len(y_val), 1))
model = Sequential([
    # LSTM layer
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),  # Regularization

    # Second LSTM layer
    LSTM(50, return_sequences=False),
    Dropout(0.2),

    # Fully connected layer
    Dense(25, activation='relu'),

    # Output layer
    Dense(1)  # Predicting a single value
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,  # Adjust as needed
    batch_size=32,  # Experiment with batch size
    verbose=1
)

results = model.evaluate(x_val, y_val, verbose=1)
print(f"Validation Loss: {results[0]}, Validation MAE: {results[1]}")

predictions = model.predict(x_val)  # Predict on validation data
model.save('stock_estimator_model.h5')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.show()

from tensorflow.keras.models import load_model
model = load_model('stock_estimator_model.h5')
test = load_processed_data('testData')
x_test = np.array(test[0]).reshape((len(test[0]), 200, 1)) 
y_test = np.array(test[1]).reshape((len(test[1]), 1))

predictions = model.predict(x_test)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

