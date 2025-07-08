import yfinance as yf
import numpy as np

def get_vol(symbol):
    hist = yf.Ticker(symbol).history(period='60d')
    if hist.shape[0] < 30:
        return None
    hist['returns'] = hist['Close'].pct_change()
    return hist['returns'][-30:].std() * np.sqrt(252)

print('ES=F:', get_vol('ES=F'))
print('SPY:', get_vol('SPY'))
print('NQ=F:', get_vol('NQ=F'))
print('QQQ:', get_vol('QQQ')) 