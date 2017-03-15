#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:28:53 2017

@author: student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


class Cointegrator(object):
  def __init__(self, x, y): # using these classic tickers instead of stock1 and stock2
    if x[-1] > y[-1]:
      self.X = np.array(quandl.get("WIKI/" + y, start_date="2010-1-1").loc[:, "Close"].tolist())
      self.Y = np.array(quandl.get("WIKI/" + x, start_date="2010-1-1").loc[:, "Close"].tolist())
    else:
      self.X = np.array(quandl.get("WIKI/" + x, start_date="2010-1-1").loc[:, "Close"].tolist())
      self.Y = np.array(quandl.get("WIKI/" + y, start_date="2010-1-1").loc[:, "Close"].tolist())
    if len(self.X) > len(self.Y):
      self.X = self.X[-len(self.Y):]
    else:
      self.Y = self.Y[-len(self.X):]
    self.diffX, self.diffY = np.diff(self.X), np.diff(self.Y)
     
   
  def adf(self):
    X = adfuller(self.diffX)[1]
    Y = adfuller(self.diffY)[1]
    if X < .05 and Y < .05:
      return True
    else:
      return False
  
  def engel_granger(self):
    adf = self.adf()
    if adf:
        if self.diffX[-1] > self.diffY[-1]:
          small = self.diffY
          large = self.diffX
        else:
          small = self.diffX
          large = self.diffY
        def test(small, large):
          modelX = sm.add_constant(small)
          model = sm.OLS(large, modelX)
          results = model.fit()
          ws = results.params
          yhat = small * ws[1] + ws[0]
          residuals = large - yhat
          resid = adfuller(residuals)
          if resid[1] > .05:
            self.cointegrated = False
          else:
            cointegrated = True
          return cointegrated
    try:
        return test(small, large)
    except NameError:
        return "Equities Not Integrated of Order 1"
test1 = Cointegrator("GS", "JPM")
print(test1.engel_granger())