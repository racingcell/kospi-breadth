import os
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go
from tqdm import tqdm
from datetime import datetime

# =========================
# CONFIG
# =========================
MARKET = "KOSPI"
START_DATE = "2023-01-01"   # calculations need history
DISPLAY_START = "2024-01-01"
OUTPUT_DIR = "docs"
DATA_DIR = "docs/data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def save_fig(fig, filename):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=20, t=40, b=30),
        xaxis=dict(range=[DISPLAY_START, None]),
    )
    fig.write_html(
        f"{OUTPUT_DIR}/{filename}",
        include_plotlyjs="cdn",
        config={"responsive": True}
    )

# =========================
# LOAD TICKERS
# =========================
tickers = fdr.StockListing(MARKET)["Code"].tolist()

prices = {}
for t in tqdm(tickers, desc="Downloading prices"):
    try:
        df = fdr.DataReader(t, START_DATE)
        prices[t] = df["Close"]
    except:
        pass

prices = pd.DataFrame(prices)
prices.index = pd.to_datetime(prices.index)

# =========================
# BREADTH (SMA)
# =========================
sma_periods = [20, 60, 120, 200]
breadth = {}

for p in sma_periods:
    sma = prices.rolling(p).mean()
    pct = (prices > sma).sum(axis=1) / prices.count(axis=1) * 100
    breadth[f"above_{p}"] = pct

breadth_df = pd.DataFrame(breadth)
breadth_df = breadth_df[breadth_df.index >= DISPLAY_START]
breadth_df.to_csv(f"{DATA_DIR}/breadth_sma.csv")

breadth_21dma = breadth_df.rolling(21).mean()
breadth_21dma.to_csv(f"{DATA_DIR}/breadth_sma_21dma.csv")

for col in breadth_df.columns:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=breadth_df.index, y=breadth_df[col],
                             name=col.replace("_", " ").upper()))
    fig.add_trace(go.Scatter(x=breadth_21dma.index, y=breadth_21dma[col],
                             name="21D SMA", line=dict(dash="dash")))
    fig.update_layout(title=f"KOSPI % Stocks Above {col.split('_')[1]}-Day SMA")
    save_fig(fig, f"breadth_{col.split('_')[1]}.html")

# =========================
# 52-WEEK HIGHS - LOWS
# =========================
high_52w = prices == prices.rolling(252).max()
low_52w = prices == prices.rolling(252).min()

hl_df = pd.DataFrame({
    "new_highs": high_52w.sum(axis=1),
    "new_lows": low_52w.sum(axis=1)
})
hl_df["net"] = hl_df["new_highs"] - hl_df["new_lows"]
hl_df = hl_df[hl_df.index >= DISPLAY_START]
hl_df.to_csv(f"{DATA_DIR}/high_low_52w.csv")

fig = go.Figure()
fig.add_bar(x=hl_df.index, y=hl_df["net"], name="52W Highs - Lows")
fig.update_layout(title="KOSPI 52-Week Highs minus Lows")
save_fig(fig, "high_low_52w.html")

# =========================
# ADVANCE DECLINE LINE
# =========================
returns = prices.diff()
adv = (returns > 0).sum(axis=1)
dec = (returns < 0).sum(axis=1)
net_adv = adv - dec
ad_line = net_adv.cumsum()

ad_df = pd.DataFrame({
    "advances": adv,
    "declines": dec,
    "net_advances": net_adv,
    "ad_line": ad_line
})
ad_df = ad_df[ad_df.index >= DISPLAY_START]
ad_df.to_csv(f"{DATA_DIR}/advance_decline.csv")

fig = go.Figure()
fig.add_trace(go.Scatter(x=ad_df.index, y=ad_df["ad_line"], name="AD Line"))
fig.update_layout(title="KOSPI Advance–Decline Line")
save_fig(fig, "advance_decline.html")

# =========================
# AI SUMMARY (OPTIONAL)
# =========================
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_KEY:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_KEY)

        latest = breadth_df.iloc[-1]

        prompt = f"""
You are a market strategist.

Write a concise daily summary of KOSPI market breadth using:
- Percent of stocks above 20, 60, 120 and 200 day moving averages
- Short-term vs long-term trend alignment
- Market participation quality

Latest breadth data:
{latest.to_dict()}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )

        summary = response.choices[0].message.content.strip()

        # Save plain text
        with open("docs/ai_summary.txt", "w") as f:
            f.write(summary)

        # Save HTML for GitHub Pages
        with open("docs/ai_summary.html", "w") as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont;
      background: #f7f7f7;
      margin: 0;
      padding: 12px;
    }}
    .box {{
      background: white;
      border-left: 4px solid #444;
      padding: 14px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="box">
    <h2>📊 Daily AI Market Breadth Summary</h2>
    <p>{summary.replace('\n', '<br>')}</p>
  </div>
</body>
</html>""")

        print("AI summary generated successfully.")

    except Exception as e:
        print("AI summary failed:", e)
