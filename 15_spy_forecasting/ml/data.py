"""
Project 15 -- SPY forecasting: market data acquisition.

Pulls real daily bars for the S&P 500 ETF (SPY) plus two macro series, via
Yahoo Finance, and caches them so a rerun is offline and reproducible.

    python ml/data.py

The modelling target is the **log return**, not the price. Prices are
non-stationary and trivially "predictable" in the sense that tomorrow's price
is approximately today's -- a model predicting price levels scores a
spectacular R-squared while knowing nothing. Returns remove that illusion, and
what is left is genuinely hard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dsfs import Provenance, fetch_via, stamp_frame

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

START = "2015-01-01"
END = "2026-09-01"

TICKERS = {
    "SPY": "S&P 500 ETF",
    "^VIX": "CBOE volatility index",
    "^TNX": "US 10-year Treasury yield",
}

MARKET = Provenance(
    slug="yahoo-spy-daily",
    name="SPY Daily Bars with VIX and 10-Year Yield",
    source_url="https://finance.yahoo.com/quote/SPY/history",
    license="Yahoo Finance terms of use — personal and research use",
    citation=(
        "Yahoo Finance (2026). Historical daily price data for SPY, ^VIX and "
        f"^TNX, {START} to {END}. Retrieved via the yfinance library."
    ),
    notes=(
        "Split- and dividend-adjusted daily closes. SPY is the S&P 500 ETF; "
        "^VIX is implied volatility; ^TNX is the 10-year Treasury yield."
    ),
)


WHY_RETURNS_NOT_PRICES = """
Forecasting a price level is the easiest way to produce an impressive-looking
and entirely worthless result.

Tomorrow's SPY close is within a fraction of a percent of today's. A model that
simply repeats today's price scores an R-squared above 0.999 against the price
series -- while containing no information whatsoever about what will happen. A
great deal of published "stock prediction" is exactly this, plotted as a
forecast line lying convincingly on top of the actual line.

This project forecasts the daily log return instead:

    r_t = ln(P_t / P_{t-1})

which is close to stationary and has no such free lunch. The consequence is
uncomfortable but correct: R-squared on returns will sit near zero, and the
random-walk baseline -- predicting exactly zero -- is genuinely hard to beat.
That is what the honest version of this problem looks like.
"""


def _download(dest: Path) -> None:
    """Fetch from Yahoo. Called only on a cache miss."""
    import yfinance as yf

    raw = yf.download(list(TICKERS), start=START, end=END,
                      auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    volume = raw["Volume"]["SPY"].rename("spy_volume")
    high = raw["High"]["SPY"].rename("spy_high")
    low = raw["Low"]["SPY"].rename("spy_low")

    frame = pd.concat([close, volume, high, low], axis=1)
    frame.index.name = "date"
    frame.to_parquet(dest)


def load() -> tuple[pd.DataFrame, Provenance]:
    """Daily bars with the target return already computed."""
    path, prov = fetch_via(MARKET, _download, "spy_vix_tnx_daily.parquet")
    df = pd.read_parquet(path)

    df = df.rename(columns={"SPY": "spy", "^VIX": "vix", "^TNX": "tnx"})
    df = df.sort_index()

    # SPY must be present; the macro series are forward-filled over their own
    # occasional gaps (holidays differ between venues) and never backfilled,
    # since backfilling would import a future value into the past.
    df = df[df["spy"].notna()].copy()
    for c in ("vix", "tnx"):
        df[c] = df[c].ffill()
    df = df.dropna(subset=["vix", "tnx"])

    df["log_return"] = np.log(df["spy"] / df["spy"].shift(1))
    df = df.dropna(subset=["log_return"])

    return df, stamp_frame(prov, df)


def summarise(df: pd.DataFrame) -> dict:
    r = df["log_return"]
    ann = np.sqrt(252)
    # Largest single-day moves, which is where any honest interval either
    # holds up or does not.
    worst = r.nsmallest(3)
    best = r.nlargest(3)
    return {
        "n_days": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "first_close": float(df["spy"].iloc[0]),
        "last_close": float(df["spy"].iloc[-1]),
        "total_return_pct": float((df["spy"].iloc[-1] / df["spy"].iloc[0] - 1) * 100),
        "ann_return_pct": float(r.mean() * 252 * 100),
        "ann_vol_pct": float(r.std() * ann * 100),
        "sharpe_buy_hold": float(r.mean() / r.std() * ann),
        "skew": float(r.skew()),
        # Fat tails: a normal distribution has excess kurtosis 0.
        "excess_kurtosis": float(r.kurtosis()),
        "worst_day_pct": float(worst.iloc[0] * 100),
        "worst_day_date": str(worst.index[0].date()),
        "best_day_pct": float(best.iloc[0] * 100),
        "best_day_date": str(best.index[0].date()),
        "pct_up_days": float((r > 0).mean() * 100),
        "max_drawdown_pct": float(_max_drawdown(df["spy"]) * 100),
        "mean_vix": float(df["vix"].mean()),
    }


def _max_drawdown(prices: pd.Series) -> float:
    curve = prices / prices.cummax()
    return float(curve.min() - 1.0)


def price_history(df: pd.DataFrame, tail: int = 500) -> list[dict]:
    """Recent closes for the forecast chart."""
    sub = df.tail(tail)
    return [
        {"date": str(idx.date()), "close": round(float(row.spy), 2),
         "vix": round(float(row.vix), 2)}
        for idx, row in zip(sub.index, sub.itertuples())
    ]


if __name__ == "__main__":
    df, prov = load()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    prov.save(ARTIFACTS / "provenance.json")
    s = summarise(df)

    print(f"\ntrading days   : {s['n_days']:,}")
    print(f"window         : {s['start']} -> {s['end']}")
    print(f"SPY close      : ${s['first_close']:.2f} -> ${s['last_close']:.2f} "
          f"({s['total_return_pct']:+.1f}%)")
    print()
    print(f"annualised ret : {s['ann_return_pct']:+.2f}%")
    print(f"annualised vol : {s['ann_vol_pct']:.2f}%")
    print(f"buy-hold Sharpe: {s['sharpe_buy_hold']:.2f}")
    print(f"max drawdown   : {s['max_drawdown_pct']:.1f}%")
    print()
    print(f"up days        : {s['pct_up_days']:.1f}%")
    print(f"excess kurtosis: {s['excess_kurtosis']:.1f}  (normal = 0; fat tails)")
    print(f"worst day      : {s['worst_day_pct']:.2f}% on {s['worst_day_date']}")
    print(f"best day       : {s['best_day_pct']:+.2f}% on {s['best_day_date']}")
