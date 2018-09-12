import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression

from time import time

import matplotlib.pyplot as plt
from matplotlib import style

import pickle
style.use('ggplot')

#################

def plot_basic(df, forecast):
    plt.plot(df.index, df.temperature, color='red', label='Known data')
    plt.plot(forecast[0], forecast[1].temperature, color='blue', label='Preciction data')
    plt.legend(loc='best')
    plt.show()


def model_svm(X_train, X_test, y_train, y_test, forecast_input):
    model = svm.SVR(kernel='rbf')

    #'C': [1, 1e2, 1e3, 5e3, 1e4]
    #'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

    C = np.linspace(1, 1e4, 5)
    gamma = np.linspace(1e-4, 1e-1, 6)

    param_grid = {'C': [1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4],
                  'gamma': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}

    t = time()
    model = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Search time %0.2fs" % (time()-t), '\n')

    print("Best estimator found by grid search:")
    print(model.best_estimator_)

    '''
    with open('svr.pickle','wb') as f:
        pickle.dump(model, f)
    '''

    pred = model.predict(X_test)
    print("R^2 = %0.4f" % metrics.r2_score(y_test, pred))
    print("MAE = %0.4f" % metrics.mean_absolute_error(y_test, pred))
    print("RMSE = %0.4f" % np.sqrt(metrics.mean_squared_error(y_test, pred)))

    forecast_output = model.predict(forecast_input)

    return forecast_output


def model_randforreg(X_train, X_test, y_train, y_test, forecast_input_scale):
    model = RandomForestRegressor(n_estimators=10000)

    t = time()
    model.fit(X_train, y_train)
    print("Search time %0.2fs" % (time()-t), '\n')

    pred = model.predict(X_test)
    print("R^2 = %0.4f" % metrics.r2_score(y_test, pred))
    print("MAE = %0.4f" % metrics.mean_absolute_error(y_test, pred))
    print("RMSE = %0.4f" % np.sqrt(metrics.mean_squared_error(y_test, pred)))

    forecast_output = model.predict(forecast_input_scale)

    return forecast_output


def forecast_plot(df, y, forecast_dates, forecast_output):
    data = plt.plot(df.drop(columns=['VAR']).index, y, color='red', label='Known data')
    #predicted_known_data = plt.plot(df.drop(columns=['VAR']).index, RF_reg.predict(X_train), color='green', label='Predicted known data')
    predicted = plt.plot(forecast_dates, forecast_output, color='blue', label='Predicted data')

    plt.xlabel('Date')
    plt.ylabel('VAR')
    plt.legend(loc='best')
    # plt.savefig('fig.png')
    plt.show()



#################
df = pd.read_csv('/Users/fempter/Documents/timeseries.csv')

df.index = pd.to_datetime(df.Date)
df = df.drop(columns=['Date', 'DateIdx'])
df = df.rename(columns={'dewpti': 'dew_point', 'hum': 'humidity', 'pressurei': 'pressure', 'tempi': 'temperature'})

df = df.drop(columns=['PCA_1'])

forecast_dates = df.loc['2017-04-24':].iloc[:96].index
forecast_input = df.loc['2017-04-24':].iloc[:96].drop(columns=['VAR'])
forecast_input_scale = preprocessing.scale(forecast_input)

df = df.loc['2017-01-01':'2017-04-23']

#plot_basic(df, [forecast_dates, forecast_input])

X = df.drop(columns=['VAR'])
#X = preprocessing.scale(X)

y = df.VAR
#y = preprocessing.scale(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

forecast = model_randforreg(X_train, X_test, y_train, y_test, forecast_input)

print(forecast)

forecast_plot(df, y, forecast_dates, forecast)
