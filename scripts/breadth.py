import FinanceDataReader as fdr
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

# =========================
# Configuration
# =========================
MARKET = "KOSPI"
START_DATE = "2010-01-01"   # long history for correct calculations
DISPLAY_START = "2024-01-01"
MA_PERIODS = [20, 60, 120, 200]

OUTPUT_DIR = "docs"

# =========================
# Helper function
# =========================
def save_fig(fig, path):
    fig.update_xaxes(range=[DISPLAY_START, None])
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        config={"responsive": True}
    )

# =========================
# Load tickers
# =========================
listing = fdr.StockListing(MARKET)
tickers = listing["Code"].tolist()

# =========================
# Download price data
# =========================
prices = []

for code in tqdm(tickers, desc="Downloading prices"):
    try:
        df = fdr.DataReader(code, START_DATE)
        prices.append(df["Close"].rename(code))
    except Exception:
        continue

prices = pd.concat(prices, axis=1).sort_index()

# =========================
# Breadth: % above SMAs
# =========================
for period in MA_PERIODS:
    sma = prices.rolling(period).mean()
    above = prices > sma
    percent_above = above.sum(axis=1) / prices.count(axis=1) * 100

    sma21 = percent_above.rolling(21).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=percent_above.index,
        y=percent_above,
        name=f"% Above {period}-Day SMA",
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=sma21.index,
        y=sma21,
        name="21-Day SMA",
        line=dict(width=2, dash="dash")
    ))

    fig.update_layout(
        title=f"KOSPI % of Stocks Above {period}-Day SMA",
        yaxis_title="Percent",
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    save_fig(fig, f"{OUTPUT_DIR}/breadth_{period}.html")

# =========================
# 52-Week Highs minus Lows
# =========================
rolling_high = prices.rolling(252).max()
rolling_low = prices.rolling(252).min()

new_highs = (prices == rolling_high).sum(axis=1)
new_lows = (prices == rolling_low).sum(axis=1)
hl_diff = new_highs - new_lows

fig_hl = go.Figure()
fig_hl.add_trace(go.Bar(
    x=hl_diff.index,
    y=hl_diff,
    name="52W Highs − Lows"
))

fig_hl.update_layout(
    title="KOSPI 52-Week Highs Minus Lows",
    yaxis_title="Net Highs",
    height=600
)

save_fig(fig_hl, f"{OUTPUT_DIR}/high_low_52w.html")

# =========================
# Advance–Decline Line
# =========================
daily_returns = prices.diff()
advances = (daily_returns > 0).sum(axis=1)
declines = (daily_returns < 0).sum(axis=1)

net_advances = advances - declines
ad_line = net_advances.cumsum()
ad_sma21 = ad_line.rolling(21).mean()

fig_ad = go.Figure()
fig_ad.add_trace(go.Scatter(
    x=ad_line.index,
    y=ad_line,
    name="Advance–Decline Line",
    line=dict(width=2)
))
fig_ad.add_trace(go.Scatter(
    x=ad_sma21.index,
    y=ad_sma21,
    name="21-Day SMA",
    line=dict(width=2, dash="dash")
))

fig_ad.update_layout(
    title="KOSPI Advance–Decline Line",
    height=700,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

save_fig(fig_ad, f"{OUTPUT_DIR}/ad_line.html")

print("✅ All charts generated successfully.")
