from enum import Enum


class ETHPriceRanges(Enum):
    pr_2000_2025 = 1
    pr_2025_2050 = 2
    pr_2050_2075 = 3
    pr_2075_2100 = 4
    pr_2100_2125 = 5
    pr_2125_2150 = 6
    pr_2150_2175 = 7
    pr_2175_2200 = 8
    pr_2200_2225 = 9
    pr_2225_2250 = 10
    pr_2250_2275 = 11
    pr_2275_2300 = 12
    pr_2300_2325 = 13
    pr_2325_2350 = 14
    pr_2350_2375 = 15
    pr_2375_2400 = 16
    pr_2400_2425 = 17
    pr_2425_2450 = 18
    pr_2450_2475 = 19
    pr_2475_2500 = 20
    pr_2500_2525 = 21
    pr_2525_2550 = 22
    pr_2550_2575 = 23
    pr_2575_2600 = 24
    pr_2600_2625 = 25
    pr_2625_2650 = 26
    pr_2650_2675 = 27
    pr_2675_2700 = 28
    pr_2700_2725 = 29
    pr_2725_2750 = 30
    pr_2750_2775 = 31
    pr_2775_2800 = 32
    pr_2800_2825 = 33
    pr_2825_2850 = 34
    pr_2850_2875 = 35
    pr_2875_2900 = 36
    pr_2900_2925 = 37
    pr_2925_2950 = 38
    pr_2950_2975 = 39
    pr_2975_3000 = 40
    pr_3000_3025 = 41
    pr_3025_3050 = 42
    pr_3050_3075 = 43
    pr_3075_3100 = 44
    pr_3100_3125 = 45
    pr_3125_3150 = 46
    pr_3150_3175 = 47
    pr_3175_3200 = 48
    pr_3200_3225 = 49
    pr_3225_3250 = 50
    pr_3250_3275 = 51
    pr_3275_3300 = 52
    pr_3300_3325 = 53
    pr_3325_3350 = 54
    pr_3350_3375 = 55
    pr_3375_3400 = 56
    pr_3400_3425 = 57
    pr_3425_3450 = 58
    pr_3450_3475 = 59
    pr_3475_3500 = 60
    pr_3500_3525 = 61
    pr_3525_3550 = 62
    pr_3550_3575 = 63
    pr_3575_3600 = 64
    pr_3600_3625 = 65
    pr_3625_3650 = 66
    pr_3650_3675 = 67
    pr_3675_3700 = 68
    pr_3700_3725 = 69
    pr_3725_3750 = 70
    pr_3750_3775 = 71
    pr_3775_3800 = 72
    pr_3800_3825 = 73
    pr_3825_3850 = 74
    pr_3850_3875 = 75
    pr_3875_3900 = 76
    pr_3900_3925 = 77
    pr_3925_3950 = 78
    pr_3950_3975 = 79
    pr_3975_4000 = 80


class ARBPriceRanges(Enum):
    pr_1_1_1_2 = 1
    pr_1_2_1_3 = 2
    pr_1_3_1_4 = 3
    pr_1_4_1_5 = 4
    pr_1_5_1_6 = 5
    pr_1_6_1_7 = 6
    pr_1_7_1_8 = 7
    pr_1_8_1_9 = 8
    pr_1_9_2_0 = 9
    pr_2_0_2_1 = 10
    pr_2_1_2_2 = 11
    pr_2_2_2_3 = 12
    pr_2_3_2_4 = 13
    pr_2_4_2_5 = 14
    pr_2_5_2_6 = 15
    pr_2_6_2_7 = 16
    pr_2_7_2_8 = 17
    pr_2_8_2_9 = 18
    pr_2_9_3_0 = 19

class LINKPriceRanges(Enum):
    pr_4_6 = 1
    pr_6_8 = 2
    pr_8_10 = 3
    pr_10_12 = 4
    pr_12_14 = 5
    pr_14_16 = 6
    pr_16_18 = 7
    pr_18_20 = 8
    pr_20_22 = 9
    pr_22_24 = 10
    pr_24_26 = 11


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

def predictions_ETH():
    """
    All of the business logic should go here. The output should be a list
    of size 7 with values from ETHPriceRanges
    """
    def price_to_range(price):
      if 2000 <= price < 2025:
        return ETHPriceRanges.pr_2000_2025
      elif 2025 <= price < 2050:
          return ETHPriceRanges.pr_2025_2050
      elif 2050 <= price < 2075:
          return ETHPriceRanges.pr_2050_2075
      elif 2075 <= price < 2100:
          return ETHPriceRanges.pr_2075_2100
      elif 2100 <= price < 2125:
          return ETHPriceRanges.pr_2100_2125
      elif 2125 <= price < 2150:
          return ETHPriceRanges.pr_2125_2150
      elif 2150 <= price < 2175:
          return ETHPriceRanges.pr_2150_2175
      elif 2175 <= price < 2200:
          return ETHPriceRanges.pr_2175_2200
      elif 2200 <= price < 2225:
          return ETHPriceRanges.pr_2200_2225
      elif 2225 <= price < 2250:
          return ETHPriceRanges.pr_2225_2250
      elif 2250 <= price < 2275:
          return ETHPriceRanges.pr_2250_2275
      elif 2275 <= price < 2300:
          return ETHPriceRanges.pr_2275_2300
      elif 2300 <= price < 2325:
          return ETHPriceRanges.pr_2300_2325
      elif 2325 <= price < 2350:
          return ETHPriceRanges.pr_2325_2350
      elif 2350 <= price < 2375:
          return ETHPriceRanges.pr_2350_2375
      elif 2375 <= price < 2400:
          return ETHPriceRanges.pr_2375_2400
      elif 2400 <= price < 2425:
          return ETHPriceRanges.pr_2400_2425
      elif 2425 <= price < 2450:
          return ETHPriceRanges.pr_2425_2450
      elif 2450 <= price < 2475:
          return ETHPriceRanges.pr_2450_2475
      elif 2475 <= price < 2500:
          return ETHPriceRanges.pr_2475_2500
      elif 2500 <= price < 2525:
          return ETHPriceRanges.pr_2500_2525
      elif 2525 <= price < 2550:
          return ETHPriceRanges.pr_2525_2550
      elif 2550 <= price < 2575:
          return ETHPriceRanges.pr_2550_2575
      elif 2575 <= price < 2600:
          return ETHPriceRanges.pr_2575_2600
      elif 2600 <= price < 2625:
          return ETHPriceRanges.pr_2600_2625
      elif 2625 <= price < 2650:
          return ETHPriceRanges.pr_2625_2650
      elif 2650 <= price < 2675:
          return ETHPriceRanges.pr_2650_2675
      elif 2675 <= price < 2700:
          return ETHPriceRanges.pr_2675_2700
      elif 2700 <= price < 2725:
          return ETHPriceRanges.pr_2700_2725
      elif 2725 <= price < 2750:
          return ETHPriceRanges.pr_2725_2750
      elif 2750 <= price < 2775:
          return ETHPriceRanges.pr_2750_2775
      elif 2775 <= price < 2800:
          return ETHPriceRanges.pr_2775_2800
      elif 2800 <= price < 2825:
          return ETHPriceRanges.pr_2800_2825
      elif 2825 <= price < 2850:
          return ETHPriceRanges.pr_2825_2850
      elif 2850 <= price < 2875:
          return ETHPriceRanges.pr_2875_2900
      elif 2875 <= price < 2900:
          return ETHPriceRanges.pr_2875_2900
      elif 2900 <= price < 2925:
          return ETHPriceRanges.pr_2900_2925
      elif 2925 <= price < 2950:
          return ETHPriceRanges.pr_2925_2950
      elif 2950 <= price < 2975:
          return ETHPriceRanges.pr_2950_2975
      elif 2975 <= price < 3000:
          return ETHPriceRanges.pr_2975_3000
      elif 3000 <= price < 3025:
          return ETHPriceRanges.pr_3000_3025
      elif 3025 <= price < 3050:
          return ETHPriceRanges.pr_3025_3050
      elif 3050 <= price < 3075:
          return ETHPriceRanges.pr_3050_3075
      elif 3075 <= price < 3100:
          return ETHPriceRanges.pr_3075_3100
      elif 3100 <= price < 3125:
          return ETHPriceRanges.pr_3100_3125
      elif 3125 <= price < 3150:
          return ETHPriceRanges.pr_3125_3150
      elif 3150 <= price < 3175:
          return ETHPriceRanges.pr_3150_3175
      elif 3175 <= price < 3200:
          return ETHPriceRanges.pr_3175_3200
      elif 3200 <= price < 3225:
          return ETHPriceRanges.pr_3200_3225
      elif 3225 <= price < 3250:
          return ETHPriceRanges.pr_3225_3250
      elif 3250 <= price < 3275:
          return ETHPriceRanges.pr_3250_3275
      elif 3275 <= price < 3300:
          return ETHPriceRanges.pr_3275_3300
      elif 3300 <= price < 3325:
          return ETHPriceRanges.pr_3300_3325
      elif 3325 <= price < 3350:
          return ETHPriceRanges.pr_3325_3350
      elif 3350 <= price < 3375:
          return ETHPriceRanges.pr_3350_3375
      elif 3375 <= price < 3400:
          return ETHPriceRanges.pr_3375_3400
      elif 3400 <= price < 3425:
          return ETHPriceRanges.pr_3400_3425 # Return None if the price does not fall into any range
      elif 3425 <= price < 3450:
        return ETHPriceRanges.pr_3425_3450
      elif 3450 <= price < 3475:
          return ETHPriceRanges.pr_3450_3475
      elif 3475 <= price < 3500:
          return ETHPriceRanges.pr_3475_3500
      elif 3500 <= price < 3525:
          return ETHPriceRanges.pr_3500_3525
      elif 3525 <= price < 3550:
          return ETHPriceRanges.pr_3525_3550
      elif 3550 <= price < 3575:
          return ETHPriceRanges.pr_3550_3575
      elif 3575 <= price < 3600:
          return ETHPriceRanges.pr_3575_3600
      elif 3600 <= price < 3625:
          return ETHPriceRanges.pr_3600_3625
      elif 3625 <= price < 3650:
          return ETHPriceRanges.pr_3625_3650
      elif 3650 <= price < 3675:
          return ETHPriceRanges.pr_3650_3675
      elif 3675 <= price < 3700:
          return ETHPriceRanges.pr_3675_3700
      elif 3700 <= price < 3725:
          return ETHPriceRanges.pr_3700_3725
      elif 3725 <= price < 3750:
          return ETHPriceRanges.pr_3725_3750
      elif 3750 <= price < 3775:
          return ETHPriceRanges.pr_3750_3775
      elif 3775 <= price < 3800:
          return ETHPriceRanges.pr_3775_3800
      elif 3800 <= price < 3825:
          return ETHPriceRanges.pr_3800_3825
      elif 3825 <= price < 3850:
          return ETHPriceRanges.pr_3825_3850
      elif 3850 <= price < 3875:
          return ETHPriceRanges.pr_3850_3875
      elif 3875 <= price < 3900:
          return ETHPriceRanges.pr_3875_3900
      elif 3900 <= price < 3925:
          return ETHPriceRanges.pr_3900_3925
      elif 3925 <= price < 3950:
          return ETHPriceRanges.pr_3925_3950
      elif 3950 <= price < 3975:
          return ETHPriceRanges.pr_3950_3975
      elif 3975 <= price < 4000:
          return ETHPriceRanges.pr_3975_4000
      else:
          return None


    df2 = yf.download("ETH-USD", start="2023-04-01", end="2024-04-16")
    X=df2['Close']
    size = int(len(X))
    train= X[0:size]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    print('Predictions for Ethereum (ETH)')
    predicted_eth=[]
    for t in range(7):
      model = ARIMA(history, order=(4,1,1))
      model_fit = model.fit()
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      history.append(yhat)
      print(price_to_range(yhat))
      predicted_eth.append(price_to_range(yhat))
      # print('%d April expected=%f' % (t+16, yhat))
    return predicted_eth 
    # return [
    #     ETHPriceRanges.pr_2300_2325,
    #     ETHPriceRanges.pr_2300_2325,
    #     ETHPriceRanges.pr_2300_2325,
    #     ETHPriceRanges.pr_2300_2325,
    #     ETHPriceRanges.pr_2300_2325,
    #     ETHPriceRanges.pr_2300_2325,
    #     ETHPriceRanges.pr_2300_2325,
    # ]


def predictions_ARB():
    """
    All of the business logic should go here. The output should be a list
    of size 7 with values from ARBPriceRanges
    """
    def price_to_range_arb(price):
      if 0.5 <= price < 0.6:
          return ARBPriceRanges.pr_0_5_0_6
      elif 0.6 <= price < 0.7:
          return ARBPriceRanges.pr_0_6_0_7
      elif 0.7 <= price < 0.8:
          return ARBPriceRanges.pr_0_7_0_8
      elif 0.8 <= price < 0.9:
          return ARBPriceRanges.pr_0_8_0_9
      elif 0.9 <= price < 1.0:
          return ARBPriceRanges.pr_0_9_1_0
      elif 1.0 <= price < 1.1:
          return ARBPriceRanges.pr_1_0_1_1
      elif 1.1 <= price < 1.2:
          return ARBPriceRanges.pr_1_1_1_2
      elif 1.2 <= price < 1.3:
          return ARBPriceRanges.pr_1_2_1_3
      elif 1.3 <= price < 1.4:
          return ARBPriceRanges.pr_1_3_1_4
      elif 1.4 <= price < 1.5:
          return ARBPriceRanges.pr_1_4_1_5
      elif 1.5 <= price < 1.6:
          return ARBPriceRanges.pr_1_5_1_6
      elif 1.6 <= price < 1.7:
          return ARBPriceRanges.pr_1_6_1_7
      elif 1.7 <= price < 1.8:
          return ARBPriceRanges.pr_1_7_1_8
      elif 1.8 <= price < 1.9:
          return ARBPriceRanges.pr_1_8_1_9
      elif 1.9 <= price < 2.0:
          return ARBPriceRanges.pr_1_9_2_0
      elif 2.0 <= price < 2.1:
          return ARBPriceRanges.pr_2_0_2_1
      elif 2.1 <= price < 2.2:
          return ARBPriceRanges.pr_2_1_2_2
      elif 2.2 <= price < 2.3:
          return ARBPriceRanges.pr_2_2_2_3
      elif 2.3 <= price < 2.4:
          return ARBPriceRanges.pr_2_3_2_4
      elif 2.4 <= price < 2.5:
          return ARBPriceRanges.pr_2_4_2_5
      elif 2.5 <= price < 2.6:
          return ARBPriceRanges.pr_2_5_2_6
      elif 2.6 <= price < 2.7:
          return ARBPriceRanges.pr_2_6_2_7
      elif 2.7 <= price < 2.8:
          return ARBPriceRanges.pr_2_7_2_8
      elif 2.8 <= price < 2.9:
          return ARBPriceRanges.pr_2_8_2_9
      elif 2.9 <= price < 3.0:
          return ARBPriceRanges.pr_2_9_3_0
      else:
          return None
  # Return None if the price does not fall into any range

    df1 = yf.download("ARB11841-USD", start="2023-04-01", end="2024-04-16")
    X=df1['Close']
    size = int(len(X))
    train= X[0:size]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    print('Predictions for Arbitrum(ARB)')
    predicted_arb=[]
    for t in range(7):
      model = ARIMA(history, order=(4,1,1))
      model_fit = model.fit()
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      history.append(yhat)
      print(price_to_range_arb(yhat))
      predicted_arb.append(price_to_range_arb(yhat))
    return predicted_arb
      # print('%d April expected=%f' % (t+15, yhat))
    # return [
    #     ARBPriceRanges.pr_180_185,
    #     ARBPriceRanges.pr_180_185,
    #     ARBPriceRanges.pr_180_185,
    #     ARBPriceRanges.pr_180_185,
    #     ARBPriceRanges.pr_180_185,
    #     ARBPriceRanges.pr_180_185,
    #     ARBPriceRanges.pr_180_185,
    # ]


def predictions_LINK():
    """
    All of the business logic should go here. The output should be a list
    of size 7 with values from LINKPriceRanges
    """
    def price_to_range_link(price):
      if 4 <= price < 6:
          return LINKPriceRanges.pr_4_6
      elif 6 <= price < 8:
          return LINKPriceRanges.pr_6_8
      elif 8 <= price < 10:
          return LINKPriceRanges.pr_8_10
      elif 10 <= price < 12:
          return LINKPriceRanges.pr_10_12
      elif 12 <= price < 14:
          return LINKPriceRanges.pr_12_14
      elif 14 <= price < 16:
          return LINKPriceRanges.pr_14_16
      elif 16 <= price < 18:
          return LINKPriceRanges.pr_16_18
      elif 18 <= price < 20:
          return LINKPriceRanges.pr_18_20
      elif 20 <= price < 22:
          return LINKPriceRanges.pr_20_22
      elif 22 <= price < 24:
          return LINKPriceRanges.pr_22_24
      elif 24 <= price < 26:
          return LINKPriceRanges.pr_24_26
      else:
          return None # Return None if the price does not fall into any range

    df = yf.download("LINK-USD", start="2023-04-01", end="2024-04-16")
    X=df['Close']
    size = int(len(X))
    train= X[0:size]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    print('Predictions for LINKCHAIN')
    predicted_link=[]
    for t in range(7):
      model = ARIMA(history, order=(1,1,1))
      model_fit = model.fit()
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      history.append(yhat)
      price_outcome=price_to_range_link(yhat)
      print(price_outcome)
      predicted_link.append(price_outcome)
    return predicted_link
      # print('%d April expected=%f' % (t+15, yhat))
    # return [
    #     LINKPriceRanges.pr_1875_1900,
    #     LINKPriceRanges.pr_1875_1900,
    #     LINKPriceRanges.pr_1875_1900,
    #     LINKPriceRanges.pr_1875_1900,
    #     LINKPriceRanges.pr_1875_1900,
    #     LINKPriceRanges.pr_1875_1900,
    #     LINKPriceRanges.pr_1875_1900,
    # ]


"""
DO NOT REMOVE
"""
preds_ETH = predictions_ETH()
assert len(preds_ETH) == 7
assert all([isinstance(val, ETHPriceRanges) for val in preds_ETH])

preds_ARB = predictions_ARB()
assert len(preds_ARB) == 7
assert all([isinstance(val, ARBPriceRanges) for val in preds_ARB])

preds_LINK = predictions_LINK()
assert len(preds_LINK) == 7
assert all([isinstance(val, LINKPriceRanges) for val in preds_LINK])
