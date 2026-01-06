import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go

# ======================
# Parameters
# ======================
MARKET = "KOSPI"
START_DATE = "2024-01-01"
MA_WINDOWS = [20, 60, 120, 200]
SMOOTH_WINDOW = 21
OUTPUT_DIR = "docs"
DISPLAY_START = "2024-01-01"


# ======================
# Load ticker list
# ======================
listing = fdr.StockListing(MARKET)
tickers = listing["Code"].tolist()

# ======================
# Download price data
# ======================
price_data = {}

for ticker in tickers:
    try:
        df = fdr.DataReader(ticker, START_DATE)
        if not df.empty:
            price_data[ticker] = df["Close"]
    except Exception:
        continue

prices = pd.DataFrame(price_data)
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

df = df[df.index >= DISPLAY_START]


# ======================
# Breadth: % above moving averages (+ 21D SMA)
# ======================
for window in MA_WINDOWS:
    sma = prices.rolling(window).mean()
    percent_above = (prices > sma).sum(axis=1) / prices.count(axis=1) * 100
    percent_sma21 = percent_above.rolling(SMOOTH_WINDOW).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=percent_above.index,
        y=percent_above,
        mode="lines",
        name="% Above MA",
        line=dict(width=1)
    ))

    fig.add_trace(go.Scatter(
        x=percent_sma21.index,
        y=percent_sma21,
        mode="lines",
        name="21-Day SMA",
        line=dict(width=2)
    ))

    fig.update_layout(
        title=f"KOSPI % of Stocks Above {window}-Day SMA",
        height=800,
        autosize = True,
        yaxis_title="Percent (%)",
        template="plotly_white"
    )
    fig.update_xaxes(range=["2024-01-01", df.index.max()])
    fig.write_html(f"{OUTPUT_DIR}/breadth_{window}.html")

# ======================
# 52-week highs minus lows (bar chart, NO SMA)
# ======================
lookback = 252

rolling_high = prices.rolling(lookback).max()
rolling_low = prices.rolling(lookback).min()

new_highs = prices.eq(rolling_high)
new_lows = prices.eq(rolling_low)

nh_nl = new_highs.sum(axis=1) - new_lows.sum(axis=1)

bar_colors = nh_nl.apply(
    lambda x: "green" if x > 0 else ("red" if x < 0 else "gray")
)

fig_nhnl = go.Figure()
fig_nhnl.add_trace(go.Bar(
    x=nh_nl.index,
    y=nh_nl,
    marker_color=bar_colors,
    name="52W Highs − Lows"
))

# ---- regime shading: 3 bars in a row
sign = nh_nl.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
streak = sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1

current_start = None
current_sign = None

for date, s, length in zip(nh_nl.index, sign, streak):
    if s == 0:
        current_start = None
        continue

    if length == 3:
        current_start = date
        current_sign = s

    if length < 3 and current_start is not None:
        fig_nhnl.add_vrect(
            x0=current_start,
            x1=date,
            fillcolor="green" if current_sign > 0 else "red",
            opacity=0.15,
            line_width=0,
            layer="below"
        )
        current_start = None

fig_nhnl.update_layout(
    title="KOSPI 52-Week Highs Minus Lows",
    yaxis_title="Number of Stocks",
    bargap=0,
    template="plotly_white"
)
fig.update_xaxes(range=["2024-01-01", df.index.max()])
fig_nhnl.write_html(f"{OUTPUT_DIR}/breadth_52w_highs_lows.html")

# ======================
# Advance–Decline Line (+ 21D SMA)
# ======================
daily_diff = prices.diff()

advances = (daily_diff > 0).sum(axis=1)
declines = (daily_diff < 0).sum(axis=1)

net_advances = advances - declines
ad_line = net_advances.cumsum()
ad_sma21 = ad_line.rolling(SMOOTH_WINDOW).mean()

fig_ad = go.Figure()

fig_ad.add_trace(go.Scatter(
    x=ad_line.index,
    y=ad_line,
    mode="lines",
    name="AD Line",
    line=dict(width=1)
))

fig_ad.add_trace(go.Scatter(
    x=ad_sma21.index,
    y=ad_sma21,
    mode="lines",
    name="21-Day SMA",
    line=dict(width=2)
))

fig_ad.update_layout(
    title="KOSPI Advance–Decline Line",
    yaxis_title="Cumulative Net Advances",
    template="plotly_white"
)
fig.update_xaxes(range=["2024-01-01", df.index.max()])
fig_ad.write_html(f"{OUTPUT_DIR}/breadth_ad_line.html")
