# app.py
import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="News Sentiment ↔ Markets (Granger)", layout="wide")
st.title("📰→📈 News Sentiment & Market Causality (Granger)")

st.markdown(
    """
This app fetches Google News RSS headlines for a keyword, scores sentiment (VADER),
pulls market data (Nifty 50, Crude Oil, USD/INR, Gold), aggregates **daily average sentiment**,
and runs **two-way Granger causality tests** (Sentiment ↔ Asset).  
Use it for **exploration** (not trading signals).
    """
)

# -----------------------------
# Data fetchers (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_google_news_rss(keyword: str, days: int = 10, end_date: datetime | None = None) -> pd.DataFrame:
    """Fetch Google News RSS headlines for last `days` and compute VADER sentiment on titles."""
    base_url = "https://news.google.com/rss/search?q="
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    query = keyword.replace(" ", "%20")
    url = f"{base_url}{query}%20when%3A{days}d&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)

    cutoff_date = start_date.date()
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for entry in feed.entries:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub_date = datetime(*entry.published_parsed[:6]).date()
            if cutoff_date <= pub_date <= end_date.date():
                title = entry.title if hasattr(entry, "title") else ""
                sentiment = analyzer.polarity_scores(title).get("compound", 0.0)
                rows.append(
                    {
                        "Date": pub_date,
                        "Title": title,
                        "Source": getattr(getattr(entry, "source", {}), "title", "") or "",
                        "Link": entry.link if hasattr(entry, "link") else "",
                        "Sentiment": sentiment,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def fetch_market_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch Close prices for the fixed asset set and return a tidy frame."""
    tickers = {
        "Nifty 50": "^NSEI",
        "Crude Oil": "CL=F",
        "USD/INR": "INR=X",
        "Gold": "GC=F",
    }
    px = yf.download(list(tickers.values()), start=start, end=end, progress=False)["Close"]
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.droplevel(0)
    # rename ticker symbols to asset names
    name_map = {v: k for k, v in tickers.items()}
    px = px.rename(columns=name_map)
    px = px.reset_index()
    px["Date"] = pd.to_datetime(px["Date"]).dt.date
    return px


# -----------------------------
# Helpers
# -----------------------------
def _granger_min_pvals(yx: pd.DataFrame, max_lag: int) -> float | np.nan:
    """
    Run grangercausalitytests on a 2-col dataframe arranged [y, x] and
    return the minimum p-value over lags from the SSR F-test.
    """
    try:
        res = grangercausalitytests(yx, maxlag=max_lag, verbose=False)
        pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
        return float(np.nanmin(pvals))
    except Exception:
        return np.nan


def granger_summary_two_way(merged_df: pd.DataFrame, assets: list[str], max_lag: int = 5) -> pd.DataFrame:
    """
    For each asset column (numeric) in `assets`, compute two-way Granger:
    Sentiment → Asset  and  Asset → Sentiment.
    Uses price levels (to match your notebook). If you want returns, compute returns first.
    """
    out = []
    base = merged_df.copy()
    # ensure numeric
    for asset in assets:
        if asset not in base.columns:
            continue

        df = base[["Sentiment", asset]].dropna()
        # need sufficient rows: rule of thumb > max_lag + 5
        if len(df) < (max_lag + 5):
            out.append(
                {
                    "Asset": asset,
                    "Min P (Sent→Asset)": np.nan,
                    "Min P (Asset→Sent)": np.nan,
                    "Obs": len(df),
                    "Note": f"Insufficient rows for max_lag={max_lag}",
                }
            )
            continue

        # IMPORTANT: For x causes y, pass columns [y, x]
        # Sentiment → Asset
        min_p_sa = _granger_min_pvals(df[[asset, "Sentiment"]], max_lag)

        # Asset → Sentiment
        min_p_as = _granger_min_pvals(df[["Sentiment", asset]], max_lag)

        out.append(
            {
                "Asset": asset,
                "Min P (Sent→Asset)": round(min_p_sa, 4) if pd.notna(min_p_sa) else np.nan,
                "Min P (Asset→Sent)": round(min_p_as, 4) if pd.notna(min_p_as) else np.nan,
                "Obs": len(df),
                "Note": "",
            }
        )
    return pd.DataFrame(out)


def plot_scaled_series(merged_df: pd.DataFrame, assets: list[str]) -> None:
    """Plot 0–1 scaled assets + raw daily sentiment (−1..1)."""
    df = merged_df.copy()
    plt.figure(figsize=(12, 6))

    for col in assets:
        if col in df.columns and df[col].notna().sum() > 0:
            cmin, cmax = df[col].min(), df[col].max()
            series = (df[col] - cmin) / (cmax - cmin) if cmax > cmin else df[col] * 0
            plt.plot(pd.to_datetime(df["Date"]), series, label=col)

    plt.plot(
        pd.to_datetime(df["Date"]),
        df["Sentiment"],
        label="Sentiment (daily avg)",
        linewidth=2,
        linestyle="--",
        color="black",
    )
    plt.title("Scaled Market Assets (0–1) & Daily Sentiment (−1..1)")
    plt.xlabel("Date")
    plt.ylabel("Scaled Value / Sentiment")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


# -----------------------------
# UI Controls
# -----------------------------
col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    time_range = st.selectbox("Time Range", ["1 Month", "6 Months", "12 Months"], index=1)
with col2:
    max_lag = st.slider("Max Lag (Granger)", 1, 12, 5, 1)
with col3:
    sample_n = st.slider("Sentiment Table Sample Rows", 5, 25, 10, 1)

keyword = st.text_input("News Keyword(s)", "TRUMP AND INDIA RELATIONS")

assets_fixed = ["Nifty 50", "Crude Oil", "USD/INR", "Gold"]
assets_selected = st.multiselect(
    "Assets to Analyze (Close prices):",
    options=assets_fixed,
    default=assets_fixed,
    help="These are fetched from Yahoo Finance and merged with daily average sentiment.",
)

run = st.button("Run Analysis", type="primary")

# -----------------------------
# Main Execution
# -----------------------------
if run:
    with st.spinner("Fetching news & market data..."):
        days_map = {"1 Month": 30, "6 Months": 180, "12 Months": 365}
        days = days_map[time_range]
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)

        news_df = fetch_google_news_rss(keyword, days=days, end_date=end_dt)
        market_df = fetch_market_data(start_dt, end_dt)

    st.success(f"Date Range: **{start_dt.date()} → {end_dt.date()}** · Articles: **{len(news_df)}**")

    # --- Sentiment table & summary ---
    st.markdown("### 📰 Sentiment Table (Sample)")
    if news_df.empty:
        st.warning("No news found for the selected keyword & window.")
    else:
        st.dataframe(news_df.head(sample_n), use_container_width=True)

        smin = news_df["Sentiment"].min()
        smax = news_df["Sentiment"].max()
        srange = smax - smin
        savg = news_df["Sentiment"].mean()

        st.markdown("### 📈 Sentiment Summary (Min / Max / Range / Avg)")
        scols = st.columns(5)
        scols[0].metric("Min", f"{smin:.4f}")
        scols[1].metric("Max", f"{smax:.4f}")
        scols[2].metric("Range", f"{srange:.4f}")
        scols[3].metric("Average", f"{savg:.4f}")
        scols[4].metric("Articles", f"{len(news_df)}")

    # --- Merge daily sentiment with market data ---
    if market_df.empty:
        st.error("Market data could not be fetched.")
        st.stop()

    # Aggregate daily sentiment
    if news_df.empty:
        daily_sent = pd.DataFrame(columns=["Date", "Sentiment"])
    else:
        tmp = news_df.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"]).dt.date
        daily_sent = tmp.groupby("Date", as_index=False)["Sentiment"].mean()

    merged = pd.merge(market_df, daily_sent, on="Date", how="left")
    merged["Sentiment"] = merged["Sentiment"].fillna(0.0)

    # Keep only selected assets + Date + Sentiment
    keep_cols = ["Date", "Sentiment"] + [c for c in assets_selected if c in merged.columns]
    merged_view = merged[keep_cols].copy()

    st.markdown("### 📊 Daily Merged View (tail)")
    st.dataframe(merged_view.tail(15), use_container_width=True)

    # --- Granger two-way summary ---
    st.markdown("### 🧪 Granger Causality Summary (lower p ⇒ stronger evidence)")
    if len(assets_selected) == 0:
        st.info("Select at least one asset to run Granger tests.")
    else:
        gsum = granger_summary_two_way(merged_view, assets_selected, max_lag=max_lag)
        st.dataframe(gsum, use_container_width=True)

    # --- Plot scaled series ---
    st.markdown("### 📉 Scaled Time Series: Assets (0–1) & Sentiment (−1..1)")
    if len(assets_selected) > 0:
        plot_scaled_series(merged_view, assets_selected)

    # --- Download buttons ---
    st.markdown("### ⬇️ Downloads")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download News Table (CSV)", data=news_df.to_csv(index=False).encode("utf-8"),
            file_name="news_sentiment.csv", mime="text/csv"
        )
    with c2:
        st.download_button(
            "Download Merged Daily (CSV)", data=merged_view.to_csv(index=False).encode("utf-8"),
            file_name="merged_daily.csv", mime="text/csv"
        )
    with c3:
        if not gsum.empty:
            st.download_button(
                "Download Granger Summary (CSV)", data=gsum.to_csv(index=False).encode("utf-8"),
                file_name="granger_summary.csv", mime="text/csv"
            )

    # --- Notes ---
    st.caption(
        "Notes: Google News RSS is a sample of headlines, not full coverage. "
        "VADER is rule-based (good for short headlines, limited financial nuance). "
        "Granger tests show predictive relationships in-sample, not true causality."
    )
