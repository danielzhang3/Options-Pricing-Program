import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

# In-memory caches
_price_cache = {}
_vol_cache = {}

# Primary ticker mappings
ticker_symbols = {
    "SPX": "^SPX",
    "ES": "ES=F",
    "MES": "MES=F",
    "NQ": "NQ=F",
    "MNQ": "MNQ=F",
    "RTY": "RTY=F",
    "YM": "YM=F",
    "MYM": "MYM=F",
    "XND": "^XND",
    "XSP": "^XSP",
    "SPXW": "^SPX", 
    "DJX": "^DJX",
}

# Fallback mappings to ETF or index proxies
fallback_map = {
    "ES=F": "SPY",
    "MES=F": "SPY",
    "NQ=F": "QQQ",
    "MNQ=F": "QQQ",
    "RTY=F": "IWM",
    "YM=F": "DIA",
    "MYM=F": "DIA",
    "SPX": "^GSPC",
}

RISK_FREE_RATE = 0.0424  # 3-month T-bill rate as of July 2, 2025

def fetch_historical_price(symbol: str, date: datetime) -> float:
    base_symbol = symbol.upper().replace('$', '')
    primary = ticker_symbols.get(base_symbol, base_symbol)
    fallback = fallback_map.get(primary)

    for symbol_try in [primary, fallback]:
        if not symbol_try:
            continue
        for delta in range(0, 7):  # Retry for the last 7 days
            target_date = date - timedelta(days=delta)
            cache_key = (symbol_try, target_date.date())
            if cache_key in _price_cache:
                return _price_cache[cache_key]
            start = (target_date - timedelta(days=5)).strftime('%Y-%m-%d')
            end = target_date.strftime('%Y-%m-%d')
            try:
                data = yf.download(symbol_try, start=start, end=end, progress=False)
                closes = data['Close'].dropna()
                closes = closes[closes.index <= end]
                if not closes.empty:
                    price = float(closes.iloc[-1])
                    _price_cache[cache_key] = price
                    return price
            except Exception as e:
                print(f"Failed to fetch {symbol_try}: {e}")
                continue

    print(f"⚠️ No data found for {symbol} (tried {primary} and {fallback}) on or before {date.date()}")
    return None

def get_30d_historical_volatility(symbol: str, end_date) -> float:
    base_symbol = symbol.upper().replace('$', '')
    primary = ticker_symbols.get(base_symbol, base_symbol)
    fallback = fallback_map.get(primary)

    if isinstance(end_date, datetime):
        end_date_str = end_date.strftime('%Y-%m-%d')
        cache_key = (primary, end_date_str)
    else:
        end_date_str = str(end_date)
        cache_key = (primary, end_date_str)

    if cache_key in _vol_cache:
        return _vol_cache[cache_key]

    for symbol_try in [primary, fallback]:
        if not symbol_try:
            continue
        try:
            ticker = yf.Ticker(symbol_try)
            hist = ticker.history(end=end_date_str, period="60d")
            if hist.shape[0] < 30:
                continue
            hist['returns'] = hist['Close'].pct_change()
            vol = hist['returns'][-30:].std() * np.sqrt(252)
            if not np.isnan(vol):
                _vol_cache[cache_key] = vol
                return vol
        except Exception as e:
            print(f"Error fetching volatility for {symbol_try}: {e}")
            continue

    print(f"⚠️ No volatility data found for {symbol} (tried {primary} and {fallback}) ending on {end_date_str}")
    return None

def get_risk_free_rate(date=None) -> float:
    return RISK_FREE_RATE
