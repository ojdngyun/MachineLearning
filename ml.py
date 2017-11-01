import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import glob
import pickle

plt.style.use('ggplot')

ADJ_OPEN = 'Adj. Open'
ADJ_HIGH = 'Adj. High'
ADJ_LOW = 'Adj. Low'
ADJ_CLOSE = 'Adj. Close'
ADJ_VOLUME = 'Adj. Volume'

HL_PCT = 'HL_PCT'
PCT_CHANGE = 'PCT_change'
LABEL = 'label'
FORECAST = 'Forecast'

df = quandl.get('WIKI/GOOGL')
df = df[[ADJ_OPEN, ADJ_HIGH, ADJ_LOW, ADJ_CLOSE, ADJ_VOLUME]]
df[HL_PCT] = (df[ADJ_HIGH] - df[ADJ_CLOSE]) / df[ADJ_CLOSE] * 100.0
df[PCT_CHANGE] = (df[ADJ_CLOSE] - df[ADJ_OPEN]) / df[ADJ_OPEN] * 100.0

df = df[[ADJ_CLOSE, HL_PCT, PCT_CHANGE, ADJ_VOLUME]]

forecast_col = ADJ_CLOSE
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print("making forecast for: " + str(forecast_out) + " days\n")

df[LABEL] = df[forecast_col].shift(-forecast_out)

# capital x as features
X = np.array(df.drop([LABEL], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
# lower case y as labels
y = np.array(df[LABEL])

# splits the data into training and testing sets
# shuffles the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = LinearRegression()
# classifier = svm.SVR()

# getting list of files in current directory
files = glob.glob("*")

if 'linearRegression.pickle' in files:
    print('loading model from disk\n')
    pickle_in = open('linearRegression.pickle', 'rb')
    classifier = pickle.load(pickle_in)
else:
    print('training model and saving to disk\n')
    # training the classifier
    classifier.fit(X_train, y_train)

    with open('linearRegression.pickle', 'wb') as f:
        pickle.dump(classifier, f)

# testing the accuracy of the model (classifier)
accuracy = classifier.score(X_test, y_test)

# making prediction for the next forecast_out days ahead
forecast_set = classifier.predict(X_lately)
print("forecast set:\n" + str(forecast_set) + "\n")
print("accuracy: " + str(accuracy) + "\n")

# initializing 'Forecast' column to have nan values
df[FORECAST] = np.nan

# last date in the data frame
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # creating the predictions for the next forecast set
    # NAN values are added for the unknown features
    # the forecast for the day [i] is added at the end
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df[ADJ_CLOSE].plot()
df[FORECAST].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
# plt.show()

