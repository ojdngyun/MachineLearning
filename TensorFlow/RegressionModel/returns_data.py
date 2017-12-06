import pandas as pd
import numpy as np


# return a tuple with 2 fields, the return for google and the S&P 500
# each of the returns are in the form of 1D array
def read_goog_sp500_data():
    # point ot where you've stored the csv file on your local machine
    goog_file = './GOOG.csv'
    sp_file = './SP_500.csv'

    goog = pd.read_csv(goog_file, sep=",", usecols=[0, 5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(sp_file, sep=",", usecols=[0, 5], names=['Date', 'SP500'], header=0)

    goog['SP500'] = sp['SP500']

    # the date object is a string, format it as a date
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')

    goog = goog.sort_values(['Date'], ascending=[True])

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]].pct_change()

    # filter out the very first row which does not have any value for returns
    xData = np.array(returns["SP500"])[2:]
    yData = np.array(returns["Goog"])[2:]

    return xData, yData
