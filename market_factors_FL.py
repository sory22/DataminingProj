'''
CCI = (TP-MATP)/(c * MD)
TP = typical price -- avg of high, low, closing price 
MATP = moving average of the typical price for N (20) periods
c = constant (0.015)
MD = avg difference of day to day typical price -- absolute value
'''

from __future__ import print_function # Python 2/3 compatibility
import boto3
import json
import decimal
import pandas as pd
import numpy as np
import csv
import json
from collections import defaultdict
import matplotlib.dates as mdates

def movingaverage(values,window):
  weigths = np.repeat(1.0, window)/window
  smas = np.convolve(values, weigths, 'valid')
  return smas # as a numpy array


def ExpMovingAverage(values, window):
  weights = np.exp(np.linspace(-1., 0., window))
  weights /= weights.sum()
  a =  np.convolve(values, weights, mode='full')[:len(values)]
  a[:window] = a[window]
  return a


def computeMACD(x, slow=26, fast=12):
  """
  compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
  return value is emaslow, emafast, macd which are len(x) arrays
  """
  emaslow = ExpMovingAverage(x, slow)
  emafast = ExpMovingAverage(x, fast)
  return emaslow, emafast, emafast - emaslow


#From https://pythonprogramming.net/advanced-matplotlib-graphing-charting-tutorial/
def rsiFunc(prices, n=20):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def bbands(stockPrices, numDays, numStd):
  series = pd.Series(stockPrices)
  pd.to_numeric(series)

  ave = pd.stats.moments.rolling_mean(series, numDays)
  std = pd.stats.moments.rolling_std(series, numDays)

  upBand = ave[19] + (std*numStd)
  lowBand = ave[19] - (std*numStd)

  return np.round(ave, 3), np.round(upBand, 3), np.round(lowBand, 3)

def getStats(adjclose, periods, std):
  m, n = 0, 0
  stockPrices = list()
  avgs, ups, lows = list(), list(), list()
  for close in adjclose:
    stockPrices.append(close)
    if m >= periods:
      avg, up, low = bbands(stockPrices[n:m], periods, std)
      avgs.append(avg)
      ups.append(up)
      lows.append(low)
      n = n + 1

      if m > len(adjclose)-1: #exit at end
        print("DONE BROKE")
        break

    m = m + 1
  return avgs, ups, lows

stockFile = "AAPL3.csv"
op, hi, lo, cl, vol, adjcl = np.loadtxt(stockFile,delimiter=',', unpack=True)
avgs, ups, lows = getStats(cl, 20, 2)
rsi = rsiFunc(adjcl, 14)
#IF MACD ABOVE SIGNAL LINE THEN BUY -- IF NOT SELL
emaslow, emafast, macd = computeMACD(adjcl)
signal = ExpMovingAverage(macd, 9)
#Stock is BUY if RSI <= 30, SELL if >= 70...

bbavg, bbup, bblow = list(), list(), list()
for avg in avgs:
  bbavg.append(avg[19])

for low in lows:
  bblow.append(low[19])

for up in ups:
  bbup.append(up[19])