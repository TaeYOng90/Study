from __future__ import annotations

import csv
import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
INITIAL_CAPITAL = 1000.0
ENTRY_NOTIONAL = 100.0
LEVERAGE = 10.0
FEE_RATE = 0.0005
SLIPPAGE = 0.0002
MAX_ENTRIES = 5
ATR_MULT_STOP = 1.5
DISTANCE_ATR_MAX = 2.5
ADX_MIN = 25.0
RETEST_WINDOWS = [3, 72]

BASE_ARCHIVE = "https://data.binance.vision/data/futures/um"
RESULTS_DIR = Path("tmp_backtest_1h/results")
CACHE_DIR = Path("tmp_backtest_1h/cache")


def month_iter(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    last = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    while cur <= last:
        yield cur
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)


def read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("ZIP has no CSV")
        raw = zf.read(names[0])
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows:
        return pd.DataFrame()
    if not rows[0][0].isdigit():
        rows = rows[1:]
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    rows = [r[:12] for r in rows if len(r) >= 6]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    needed = ["open_time", "open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df:
            raise ValueError(f"missing {c}")
    return df[needed]


def download(url: str, session: requests.Session) -> bytes | None:
    for attempt in range(4):
        r = session.get(url, timeout=45)
        if r.status_code == 200:
            return r.content
        if r.status_code == 404:
            return None
        if r.status_code in (418, 429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    return None


def fetch_data() -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "btc-backtest-research/1.0"})
    now = datetime.now(timezone.utc)
    previous_month_last_day = datetime(now.year, now.month, 1, tzinfo=timezone.utc) - timedelta(days=1)
    parts: list[pd.DataFrame] = []

    for month in month_iter(START, previous_month_last_day):
        name = f"{SYMBOL}-{INTERVAL}-{month:%Y-%m}.zip"
        cache = CACHE_DIR / name
        if cache.exists():
            content = cache.read_bytes()
        else:
            url = f"{BASE_ARCHIVE}/monthly/klines/{SYMBOL}/{INTERVAL}/{name}"
            content = download(url, session)
            if content is None:
                continue
            cache.write_bytes(content)
        part = read_zip_csv(content)
        if not part.empty:
            parts.append(part)

    current_month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    day = current_month_start
    yesterday = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
    while day <= yesterday:
        name = f"{SYMBOL}-{INTERVAL}-{day:%Y-%m-%d}.zip"
        cache = CACHE_DIR / name
        if cache.exists():
            content = cache.read_bytes()
        else:
            url = f"{BASE_ARCHIVE}/daily/klines/{SYMBOL}/{INTERVAL}/{name}"
            content = download(url, session)
            if content is None:
                day += timedelta(days=1)
                continue
            cache.write_bytes(content)
        part = read_zip_csv(content)
        if not part.empty:
            parts.append(part)
        day += timedelta(days=1)

    if not parts:
        raise RuntimeError("No Binance public data downloaded")

    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (
        df[["timestamp", "open", "high", "low", "close", "volume"]]
        .dropna()
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df = df[df["timestamp"] >= pd.Timestamp(START)].reset_index(drop=True)
    completed_before = pd.Timestamp(datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0))
    df = df[df["timestamp"] < completed_before].reset_index(drop=True)
    expected = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq="1h", tz="UTC")
    missing = len(expected.difference(pd.DatetimeIndex(df["timestamp"])))
    print(f"Downloaded {len(df)} bars, missing hours: {missing}")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma200"] = close.rolling(200).mean()
    out["bb_mid"] = close.rolling(220).mean()
    out["bb_std"] = close.rolling(220).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + 2.0 * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - 2.0 * out["bb_std"]

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=out.index)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_sm = plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    minus_sm = minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100.0 * plus_sm / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_sm / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["atr14"] = atr
    out["adx14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["ma200_prev"] = out["ma200"].shift(1)
    return out


@dataclass
class Position:
    side: int
    entry_time: pd.Timestamp
    cycle_start_cash: float
    initial_atr: float
    qty: float = 0.0
    avg_entry: float = 0.0
    entries: int = 0
    entry_notional: float = 0.0
    last_add_price: float = 0.0
    stop: float = 0.0
    tp: float = 0.0
    partial_taken: bool = False
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0
    entry_fees: float = 0.0
    exit_fees: float = 0.0
    realized_gross: float = 0.0


def buy_price(px: float) -> float:
    return px * (1.0 + SLIPPAGE)


def sell_price(px: float) -> float:
    return px * (1.0 - SLIPPAGE)


def entry_fill(side: int, px: float) -> float:
    return buy_price(px) if side == 1 else sell_price(px)


def exit_fill(side: int, px: float) -> float:
    return sell_price(px) if side == 1 else buy_price(px)


def trend_ok(row: pd.Series, side: int) -> bool:
    required = ["ma20", "ma60", "ma200", "bb_upper", "bb_lower", "atr14", "adx14", "ma200_prev"]
    if any(pd.isna(row[c]) for c in required):
        return False
    if row["atr14"] <= 0 or row["adx14"] < ADX_MIN:
        return False
    if side == 1:
        return (
            row["ma20"] > row["ma60"] > row["ma200"]
            and row["ma200"] > row["ma200_prev"]
            and (row["close"] - row["ma20"]) <= DISTANCE_ATR_MAX * row["atr14"]
        )
    return (
        row["ma20"] < row["ma60"] < row["ma200"]
        and row["ma200"] < row["ma200_prev"]
        and (row["ma20"] - row["close"]) <= DISTANCE_ATR_MAX * row["atr14"]
    )


def breakout(row: pd.Series, side: int) -> bool:
    if not trend_ok(row, side):
        return False
    return row["close"] > row["bb_upper"] if side == 1 else row["close"] < row["bb_lower"]


def retest(row: pd.Series, side: int) -> bool:
    if not trend_ok(row, side):
        return False
    if side == 1:
        return row["low"] <= row["bb_upper"] and row["close"] > row["bb_upper"]
    return row["high"] >= row["bb_lower"] and row["close"] < row["bb_lower"]


def update_risk_levels(pos: Position):
    dist = ATR_MULT_STOP * pos.initial_atr
    if pos.side == 1:
        pos.stop = pos.avg_entry - dist
        pos.tp = pos.avg_entry + dist
    else:
        pos.stop = pos.avg_entry + dist
        pos.tp = pos.avg_entry - dist


def add_entry(pos: Position, raw_px: float, cash: float) -> float:
    fill = entry_fill(pos.side, raw_px)
    qty = ENTRY_NOTIONAL / fill
    old_cost = pos.avg_entry * pos.qty
    pos.qty += qty
    pos.avg_entry = (old_cost + fill * qty) / pos.qty
    pos.entries += 1
    pos.entry_notional += ENTRY_NOTIONAL
    pos.last_add_price = fill
    fee = ENTRY_NOTIONAL * FEE_RATE
    pos.entry_fees += fee
    cash -= fee
    update_risk_levels(pos)
    return cash


def close_qty(pos: Position, qty: float, raw_px: float, cash: float) -> tuple[float, float, float]:
    fill = exit_fill(pos.side, raw_px)
    gross = (fill - pos.avg_entry) * qty * pos.side
    notional = abs(fill * qty)
    fee = notional * FEE_RATE
    pos.realized_gross += gross
    pos.exit_fees += fee
    cash += gross - fee
    pos.qty -= qty
    if pos.qty < 1e-12:
        pos.qty = 0.0
    return cash, fill, gross - fee


def mark_to_market(pos: Position | None, close: float) -> float:
    if pos is None or pos.qty <= 0:
        return 0.0
    return (close - pos.avg_entry) * pos.qty * pos.side


def adverse_exit_level(pos: Position, ma20: float) -> float:
    levels = [pos.stop]
    if not pd.isna(ma20):
        levels.append(ma20)
    return max(levels) if pos.side == 1 else min(levels)


def level_hit(row: pd.Series, side: int, level: float, adverse: bool) -> tuple[bool, float]:
    op = float(row["open"])
    hi = float(row["high"])
    lo = float(row["low"])
    if side == 1:
        if adverse:
            if op <= level:
                return True, op
            return (lo <= level), level
        if op >= level:
            return True, op
        return (hi >= level), level
    if adverse:
        if op >= level:
            return True, op
        return (hi >= level), level
    if op <= level:
        return True, op
    return (lo <= level), level


def run_backtest(df: pd.DataFrame, retest_window: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cash = INITIAL_CAPITAL
    pos: Position | None = None
    pending: dict | None = None
    scheduled: dict | None = None
    trades: list[dict] = []
    equity_rows: list[dict] = []
    max_position_notional = 0.0
    max_margin = 0.0
    max_entries = 0
    global_worst_adverse = 0.0

    first_valid = df["bb_upper"].first_valid_index()
    start_idx = (int(first_valid) if first_valid is not None else 0) + 1

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]

        if scheduled is not None and scheduled["index"] == i:
            if scheduled["kind"] == "entry" and pos is None:
                side = scheduled["side"]
                atr = float(scheduled["atr"])
                pos = Position(side=side, entry_time=ts, cycle_start_cash=cash, initial_atr=atr)
                cash = add_entry(pos, float(row["open"]), cash)
                pending = None
            elif scheduled["kind"] == "add" and pos is not None and not pos.partial_taken:
                cash = add_entry(pos, float(row["open"]), cash)
            scheduled = None

        if pos is not None:
            if pos.side == 1:
                favorable = (float(row["high"]) / pos.avg_entry - 1.0) * 100.0
                adverse = (float(row["low"]) / pos.avg_entry - 1.0) * 100.0
            else:
                favorable = (1.0 - float(row["low"]) / pos.avg_entry) * 100.0
                adverse = (1.0 - float(row["high"]) / pos.avg_entry) * 100.0
            pos.max_favorable_pct = max(pos.max_favorable_pct, favorable)
            pos.max_adverse_pct = min(pos.max_adverse_pct, adverse)
            global_worst_adverse = min(global_worst_adverse, adverse)

            adv_level = adverse_exit_level(pos, float(row["ma20"]))
            hit_adv, adv_px = level_hit(row, pos.side, adv_level, adverse=True)
            if hit_adv:
                qty = pos.qty
                cash, exit_px, _ = close_qty(pos, qty, adv_px, cash)
                cycle_net = cash - pos.cycle_start_cash
                trades.append({
                    "entry_time": pos.entry_time,
                    "exit_time": ts,
                    "side": "long" if pos.side == 1 else "short",
                    "entries": pos.entries,
                    "entry_notional": pos.entry_notional,
                    "avg_entry": pos.avg_entry,
                    "exit_price": exit_px,
                    "partial_taken": pos.partial_taken,
                    "net_pnl": cycle_net,
                    "return_on_max_margin_pct": cycle_net / (pos.entry_notional / LEVERAGE) * 100.0,
                    "max_favorable_pct": pos.max_favorable_pct,
                    "max_adverse_pct": pos.max_adverse_pct,
                    "reason": "ma20_or_stop",
                    "entry_fees": pos.entry_fees,
                    "exit_fees": pos.exit_fees,
                })
                pos = None
                scheduled = None
            elif not pos.partial_taken:
                hit_tp, tp_px = level_hit(row, pos.side, pos.tp, adverse=False)
                if hit_tp:
                    qty = pos.qty * 0.5
                    cash, _, _ = close_qty(pos, qty, tp_px, cash)
                    pos.partial_taken = True
                    pos.stop = pos.avg_entry
                    scheduled = None

        if pos is not None:
            max_position_notional = max(max_position_notional, pos.entry_notional)
            max_margin = max(max_margin, pos.entry_notional / LEVERAGE)
            max_entries = max(max_entries, pos.entries)
            if scheduled is None and not pos.partial_taken and pos.entries < MAX_ENTRIES and i + 1 < len(df):
                threshold = pos.last_add_price + pos.initial_atr if pos.side == 1 else pos.last_add_price - pos.initial_atr
                if (pos.side == 1 and row["close"] >= threshold) or (pos.side == -1 and row["close"] <= threshold):
                    scheduled = {"kind": "add", "index": i + 1}

        if pos is None:
            if pending is not None:
                if i > pending["expiry"] or not trend_ok(row, pending["side"]):
                    pending = None
                elif i > pending["breakout_index"] and retest(row, pending["side"]) and i + 1 < len(df):
                    scheduled = {"kind": "entry", "index": i + 1, "side": pending["side"], "atr": float(row["atr14"])}
                    pending = None

            if pending is None and scheduled is None:
                long_sig = breakout(row, 1)
                short_sig = breakout(row, -1)
                if long_sig and not short_sig:
                    pending = {"side": 1, "breakout_index": i, "expiry": i + retest_window}
                elif short_sig and not long_sig:
                    pending = {"side": -1, "breakout_index": i, "expiry": i + retest_window}

        unrealized = mark_to_market(pos, float(row["close"]))
        equity_rows.append({
            "timestamp": ts,
            "cash": cash,
            "unrealized": unrealized,
            "equity": cash + unrealized,
            "side": 0 if pos is None else pos.side,
            "entries": 0 if pos is None else pos.entries,
            "position_notional": 0.0 if pos is None else pos.entry_notional,
        })

    if pos is not None:
        row = df.iloc[-1]
        qty = pos.qty
        cash, exit_px, _ = close_qty(pos, qty, float(row["close"]), cash)
        cycle_net = cash - pos.cycle_start_cash
        trades.append({
            "entry_time": pos.entry_time,
            "exit_time": row["timestamp"],
            "side": "long" if pos.side == 1 else "short",
            "entries": pos.entries,
            "entry_notional": pos.entry_notional,
            "avg_entry": pos.avg_entry,
            "exit_price": exit_px,
            "partial_taken": pos.partial_taken,
            "net_pnl": cycle_net,
            "return_on_max_margin_pct": cycle_net / (pos.entry_notional / LEVERAGE) * 100.0,
            "max_favorable_pct": pos.max_favorable_pct,
            "max_adverse_pct": pos.max_adverse_pct,
            "reason": "end_of_data",
            "entry_fees": pos.entry_fees,
            "exit_fees": pos.exit_fees,
        })
        equity_rows[-1].update({"cash": cash, "unrealized": 0.0, "equity": cash, "side": 0, "entries": 0, "position_notional": 0.0})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown_usdt"] = equity_df["equity"] - equity_df["peak"]
    equity_df["drawdown_pct"] = equity_df["drawdown_usdt"] / equity_df["peak"] * 100.0

    wins = trades_df[trades_df["net_pnl"] > 0] if not trades_df.empty else trades_df
    losses = trades_df[trades_df["net_pnl"] <= 0] if not trades_df.empty else trades_df
    gross_profit = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["net_pnl"].sum()) if not losses.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else math.inf

    mdd_idx = equity_df["drawdown_usdt"].idxmin()
    mdd_row = equity_df.loc[mdd_idx]
    peak_idx = equity_df.loc[:mdd_idx, "equity"].idxmax()

    summary = {
        "retest_window_bars": retest_window,
        "data_start": str(df["timestamp"].iloc[0]),
        "data_end": str(df["timestamp"].iloc[-1]),
        "bars": int(len(df)),
        "strategy_start": str(df["timestamp"].iloc[start_idx]),
        "initial_capital_usdt": INITIAL_CAPITAL,
        "notional_per_entry_usdt": ENTRY_NOTIONAL,
        "leverage": LEVERAGE,
        "ma_lengths": [20, 60, 200],
        "bb_length": 220,
        "bb_std": 2.0,
        "adx_length": 14,
        "adx_min": ADX_MIN,
        "atr_length": 14,
        "distance_from_ma20_max_atr": DISTANCE_ATR_MAX,
        "stop_atr": ATR_MULT_STOP,
        "partial_take_profit_r": 1.0,
        "max_entries": MAX_ENTRIES,
        "fee_each_side_pct": FEE_RATE * 100,
        "slippage_each_side_pct": SLIPPAGE * 100,
        "funding_included": False,
        "closed_cycles": int(len(trades_df)),
        "winning_cycles": int((trades_df["net_pnl"] > 0).sum()) if not trades_df.empty else 0,
        "losing_cycles": int((trades_df["net_pnl"] <= 0).sum()) if not trades_df.empty else 0,
        "win_rate_pct": float((trades_df["net_pnl"] > 0).mean() * 100.0) if not trades_df.empty else 0.0,
        "net_profit_usdt": float(cash - INITIAL_CAPITAL),
        "final_equity_usdt": float(cash),
        "profit_factor": float(pf),
        "mdd_usdt": float(mdd_row["drawdown_usdt"]),
        "mdd_pct": float(mdd_row["drawdown_pct"]),
        "mdd_peak_time": str(equity_df.loc[peak_idx, "timestamp"]),
        "mdd_trough_time": str(mdd_row["timestamp"]),
        "minimum_equity_usdt": float(equity_df["equity"].min()),
        "max_position_notional_usdt": float(max_position_notional),
        "max_initial_margin_usdt": float(max_margin),
        "max_open_entries": int(max_entries),
        "worst_intraday_adverse_move_pct": float(global_worst_adverse),
        "long_cycles": int((trades_df["side"] == "long").sum()) if not trades_df.empty else 0,
        "short_cycles": int((trades_df["side"] == "short").sum()) if not trades_df.empty else 0,
        "long_win_rate_pct": float((trades_df.loc[trades_df["side"] == "long", "net_pnl"] > 0).mean() * 100.0) if (not trades_df.empty and (trades_df["side"] == "long").any()) else 0.0,
        "short_win_rate_pct": float((trades_df.loc[trades_df["side"] == "short", "net_pnl"] > 0).mean() * 100.0) if (not trades_df.empty and (trades_df["side"] == "short").any()) else 0.0,
        "best_cycle_usdt": float(trades_df["net_pnl"].max()) if not trades_df.empty else 0.0,
        "worst_cycle_usdt": float(trades_df["net_pnl"].min()) if not trades_df.empty else 0.0,
        "average_cycle_usdt": float(trades_df["net_pnl"].mean()) if not trades_df.empty else 0.0,
        "median_cycle_usdt": float(trades_df["net_pnl"].median()) if not trades_df.empty else 0.0,
        "suggested_cash_floor_usdt": float(max_margin + abs(mdd_row["drawdown_usdt"]) + 100.0),
    }
    return summary, trades_df, equity_df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = add_indicators(fetch_data())
    df.to_csv(RESULTS_DIR / "BTCUSDT_1h_indicators.csv", index=False)

    summaries = []
    for window in RETEST_WINDOWS:
        summary, trades, equity = run_backtest(df, window)
        summaries.append(summary)
        label = f"retest_{window}bars"
        with open(RESULTS_DIR / f"summary_{label}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        trades.to_csv(RESULTS_DIR / f"trades_{label}.csv", index=False)
        equity.to_csv(RESULTS_DIR / f"equity_{label}.csv", index=False)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    compare_cols = [
        "retest_window_bars", "closed_cycles", "win_rate_pct", "net_profit_usdt",
        "profit_factor", "mdd_usdt", "mdd_pct", "max_position_notional_usdt",
        "max_initial_margin_usdt", "max_open_entries", "worst_intraday_adverse_move_pct",
        "long_cycles", "long_win_rate_pct", "short_cycles", "short_win_rate_pct",
        "best_cycle_usdt", "worst_cycle_usdt", "suggested_cash_floor_usdt",
    ]
    pd.DataFrame(summaries)[compare_cols].to_csv(RESULTS_DIR / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
