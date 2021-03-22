import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas_datareader import data as wb
from scipy.stats import norm
import statistics as stat
import yfinance as yf

#<----------SETTING THE PAGE PARAMETERS----------->
st.set_page_config(
    page_title = "Monte Carlo Simulator",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="expanded",
)

#<----------HEADING PAGE TITLE AND DESCRIPTION------------>
st.title('Brownian Monte Carlo Simulator')
st.write("""
Created By: Dmitriy Rogozhnikov

[LinkedIn](https://www.linkedin.com/in/dmitriy-rogozhnikov/)

[GitHub](https://github.com/DmitriyRogo)
""")

with st.beta_expander('Monte Carlo - About', expanded=True):
    st.write("""
    The Monte Carlo is a widely used tool to solve a variety of problems ranging from numerical integration to optimization
    of financial portfolios.

    It's an incredible tool that used across various industries. 

    The purpose of this application is calculate the probable outcomes of a given
    security using the Monte Carlo method.

    We will manipulate the number of scenarios and days we are looking to illustrate given our equity.

    This is a basic Monte Carlo simulator that utilizes Brownian motion to estimate probable rates of return. 

    Brownian motion has two main driving components. 

    1. Drift - The different directions that rates of return have had in the past.

    2. Volatility - Utilizing historical volatility and multiplying it by a standard variable.

    Using these components we can compute the daily return of any given security.

    We will run a number of simulations to simulate future trading days and the impact it will have on the portfolio. 
    """)

#<----------MONTE CARLO SIDEBAR INPUTS----------->
st.sidebar.title("Settings")

#<----------SELECTING A VALID TICKER FOR THE MONTE CARLO SIMULATION---------->
ticker = st.sidebar.text_input("Input a Ticker", value="SPY")

#<----------SELECTING A STARTING DATE FOR CALCULATING THE VOLATILITY AND DRIFT COMPONENTS------>
st.sidebar.write("""
The start date is our basis for how far back we want to collect historical data to compute our volatility and drift.

The end date will always be today's date. 
""")
startDate = st.sidebar.date_input("Historical Start Date", datetime.date(2010,1,1))

#<----------SELECTING NUMBER OF DAYS WE ARE LOOKING TO FORECAST----------->
intDays = st.sidebar.number_input("Number of Future Days to Simulate", min_value=5, max_value=None, value=50) + 1

#<----------SELECTING THE NUMBER OF SIMULATIONS TO RUN-------------------->
intTrials = st.sidebar.number_input("Total Number of Simulations to Run", min_value=5, max_value=None, value=100)

#<----------SETTING THE NUMBER OF TOTAL SHARES INVESTED WITHIN THE FUND----------->
numShares = st.sidebar.number_input("Number of " + ticker + " Shares Held", min_value=0, max_value=None, value=10)

#<----------FULL NAME OF FUND----------->
fullName = yf.Ticker(ticker).info['longName']

#<--------IMPORTING DATA FROM YAHOO FINANCE------------>
data = pd.DataFrame()
data[ticker] = wb.DataReader(ticker, data_source = 'yahoo',
start = startDate)['Close']

#<-------COMPUTING LOG RETURN-------->
log_return = np.log(1 + data.pct_change())
simple_return = (data/data.shift(1)-1)

#<-------CALCULATING DRIFT------>
u = log_return.mean()
var = log_return.var()
drift = u - (0.5 * var)
stdev = log_return.std()
Z = norm.ppf(np.random.rand(intDays, intTrials))
daily_returns = np.exp(drift.values + stdev.values * Z)

#<----WILL ADD FEATURE FOR ADVANCED SETTINGS TO MANIPULATE STANDARD DEVIATION AND MEAN------>
# st.sidebar.subheader("Advanced Settings")
# newstdev = st.sidebar.number_input("Standard Deviation", value=stdev.item(), format="%.4f")

#<-------CALCULATING STOCK PRICE-------->
price_paths = np.zeros_like(daily_returns)
price_paths[0] = data.iloc[-1]
for t in range(1, intDays):
    price_paths[t] = price_paths[t-1]*daily_returns[t]
    endValue = numShares * price_paths[t]

with st.beta_expander('Monte Carlo - Results', expanded=True):
    st.write("""
    Standard Deviation: {}

    Mean: {}

    Variance: {}

    Drift: {}
    """.format(stdev.item(), u.item(), var.item(), drift.item()), format="%.4f")

    #<-----PLOT HISTORICAL DATA------>
    st.subheader("Historical Closing Price for " + fullName)
    tickerFigure = plt.figure(figsize=(7,3))
    plt.plot(data)
    plt.xlabel("Date")
    plt.ylabel(ticker + " Price (USD)")
    st.pyplot(tickerFigure)

    #<-----PLOTTING HISTORICAL RETURNS HISTOGRAM----->
    st.subheader("Historical Frequency of Daily Returns")
    tickerHisto = plt.figure(figsize=(7,3))
    sns.distplot(log_return.iloc[1:])
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    st.pyplot(tickerHisto)

    #<-----PLOTTING MONTE CARLO CHART RESULTS------>
    st.subheader("Monte Carlo Results for " + fullName)
    mcFigure = plt.figure(figsize=(7,4))
    plt.plot(price_paths)
    plt.xlabel("# of Days Into Future")
    plt.ylabel(ticker + " Price (USD)")
    st.pyplot(mcFigure)
    
    #<-----PLOTTING MONTE CARLO HISTOGRAM RESULTS----->
    st.subheader("Density of Terminal Monte Carlo Values")
    mcHisto = plt.figure(figsize=(7,3))
    sns.distplot(pd.DataFrame(price_paths).iloc[-1])
    plt.xlabel("Price After {} Days".format(intDays-1))
    st.pyplot(mcHisto)

    #Plotting Portfolio Value Results
    portMax = max(endValue)
    portMedian = stat.median(endValue)
    portMin = min(endValue)

    st.subheader("Portfolio Results")
    st.write("Maximum Ending Portfolio Value: ${:,.2f}".format(portMax))
    st.write("Median Ending Portfolio Value: ${:,.2f}".format(portMedian))
    st.write("Minimum Ending Portfolio Value: ${:,.2f}".format(portMin))

