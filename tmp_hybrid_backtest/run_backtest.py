from __future__ import annotations

import io
import json
import math
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
START_MONTH = date(2020, 1, 1)
TODAY_UTC = datetime.now(timezone.utc).date()
LAST_COMPLETE_DAY = TODAY_UTC - timedelta(days=1)

INITIAL_CAPITAL = 1000.0
NOTIONAL = 100.0
LEVERAGE = 10.0
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002

D_MA_FAST = 20
D_MA_MID = 60
D_MA_SLOW = 200
D_BB_LEN = 220
D_BB_STD = 2.0
D_ATR_LEN = 14
D_ADX_LEN = 14
D_ADX_MIN = 25.0
D_MAX_DISTANCE_ATR = 2.5

H_EMA_LEN = 20
H_ATR_LEN = 14
RETEST_WINDOW_HOURS = 72
STOP_ATR = 1.5
PARTIAL_R = 1.0

OUT = Path("tmp_hybrid_backtest/results")
OUT.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def month_iter(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = date(cur.year + (cur.month == 12), 1 if cur.month == 12 else cur.month + 1, 1)


def get_bytes(url: str, retries: int = 5) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=90, headers={"User-Agent": "btc-hybrid-backtest/1.0"})
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return response.content
        except FileNotFoundError:
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise last


def parse_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("ZIP contains no CSV")
        raw = archive.read(names[0])

    frame = pd.read_csv(io.BytesIO(raw), header=None)
    if len(frame.columns) != 12:
        frame = pd.read_csv(io.BytesIO(raw)).iloc[:, :12]
        frame.columns = COLUMNS
    else:
        first = str(frame.iloc[0, 0]).lower()
        if "open" in first:
            frame = frame.iloc[1:].reset_index(drop=True)
        frame.columns = COLUMNS
    return frame


def normalize_timestamp(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    # Binance archives may use milliseconds or microseconds depending on vintage.
    millis = np.where(values > 100_000_000_000_000, values / 1000.0, values)
    return pd.to_datetime(millis, unit="ms", utc=True, errors="coerce")


def download_hourly() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    current_month = date(LAST_COMPLETE_DAY.year, LAST_COMPLETE_DAY.month, 1)
    previous_month_end = current_month - timedelta(days=1)

    for month in month_iter(START_MONTH, date(previous_month_end.year, previous_month_end.month, 1)):
        ym = month.strftime("%Y-%m")
        url = (
            f"https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ym}.zip"
        )
        try:
            frames.append(parse_zip(get_bytes(url)))
        except FileNotFoundError:
            print("missing monthly", url)

    day = current_month
    while day <= LAST_COMPLETE_DAY:
        ds = day.isoformat()
        url = (
            f"https://data.binance.vision/data/futures/um/daily/klines/"
            f"{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ds}.zip"
        )
        try:
            frames.append(parse_zip(get_bytes(url)))
        except FileNotFoundError:
            print("missing daily", url)
        day += timedelta(days=1)

    if not frames:
        raise RuntimeError("No hourly data downloaded")

    frame = pd.concat(frames, ignore_index=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["time"] = normalize_timestamp(frame["open_time"])
    frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    frame = frame[frame["time"].dt.date <= LAST_COMPLETE_DAY].reset_index(drop=True)
    frame[["time", "open", "high", "low", "close", "volume"]].to_csv(
        OUT / "BTCUSDT_1h.csv", index=False
    )
    return frame[["time", "open", "high", "low", "close", "volume"]].copy()


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def wilder(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def add_atr_adx(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    output = frame.copy()
    tr = true_range(output)
    up_move = output["high"].diff()
    down_move = -output["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=output.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=output.index)
    atr = wilder(tr, length)
    plus_di = 100 * wilder(plus_dm, length) / atr.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, length) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    output["atr"] = atr
    output["adx"] = wilder(dx, length)
    return output


def build_indicators(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy().set_index("time")
    daily = hourly.resample("1D", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        rows=("close", "count"),
    )
    daily = daily[daily["rows"] == 24].drop(columns="rows").reset_index()
    daily = add_atr_adx(daily, D_ATR_LEN)
    daily["ma20"] = daily["close"].rolling(D_MA_FAST).mean()
    daily["ma60"] = daily["close"].rolling(D_MA_MID).mean()
    daily["ma200"] = daily["close"].rolling(D_MA_SLOW).mean()
    daily["bb_mid"] = daily["close"].rolling(D_BB_LEN).mean()
    daily_std = daily["close"].rolling(D_BB_LEN).std(ddof=0)
    daily["bb_upper"] = daily["bb_mid"] + D_BB_STD * daily_std
    daily["bb_lower"] = daily["bb_mid"] - D_BB_STD * daily_std
    daily["long_signal"] = (
        (daily["close"] > daily["bb_upper"])
        & (daily["ma20"] > daily["ma60"])
        & (daily["ma60"] > daily["ma200"])
        & (daily["ma200"] > daily["ma200"].shift(1))
        & (daily["adx"] >= D_ADX_MIN)
        & ((daily["close"] - daily["ma20"]) <= D_MAX_DISTANCE_ATR * daily["atr"])
    )
    daily["short_signal"] = (
        (daily["close"] < daily["bb_lower"])
        & (daily["ma20"] < daily["ma60"])
        & (daily["ma60"] < daily["ma200"])
        & (daily["ma200"] < daily["ma200"].shift(1))
        & (daily["adx"] >= D_ADX_MIN)
        & ((daily["ma20"] - daily["close"]) <= D_MAX_DISTANCE_ATR * daily["atr"])
    )

    hourly = hourly.reset_index()
    hourly = add_atr_adx(hourly, H_ATR_LEN)
    hourly["ema20"] = hourly["close"].ewm(span=H_EMA_LEN, adjust=False, min_periods=H_EMA_LEN).mean()
    hourly["prev_high"] = hourly["high"].shift(1)
    hourly["prev_low"] = hourly["low"].shift(1)

    # Map the most recently completed daily candle to each hourly bar.
    map_daily = daily[["time", "ma20"]].copy()
    map_daily["available_from"] = map_daily["time"] + pd.Timedelta(days=1)
    mapped = pd.merge_asof(
        hourly.sort_values("time"),
        map_daily[["available_from", "ma20"]].sort_values("available_from"),
        left_on="time",
        right_on="available_from",
        direction="backward",
    ).rename(columns={"ma20": "prev_daily_ma20"})
    mapped = mapped.drop(columns=["available_from"])

    daily.to_csv(OUT / "daily_indicators.csv", index=False)
    mapped.to_csv(OUT / "hourly_indicators.csv", index=False)
    return mapped, daily


@dataclass
class Setup:
    side: int
    start: pd.Timestamp
    expiry: pd.Timestamp
    band: float
    signal_day: pd.Timestamp


@dataclass
class Position:
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    original_qty: float
    qty: float
    atr: float
    stop: float
    target: float
    entry_fee: float
    realized_parts: float = 0.0
    exit_fees: float = 0.0
    partial_done: bool = False
    setup_day: pd.Timestamp | None = None


def exit_fill(reference: float, side: int) -> float:
    return reference * (1 - SLIPPAGE if side == 1 else 1 + SLIPPAGE)


def entry_fill(reference: float, side: int) -> float:
    return reference * (1 + SLIPPAGE if side == 1 else 1 - SLIPPAGE)


def run_backtest(hourly: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    setups: list[Setup] = []
    for row in daily.itertuples(index=False):
        if bool(row.long_signal):
            start = row.time + pd.Timedelta(days=1)
            setups.append(Setup(1, start, start + pd.Timedelta(hours=RETEST_WINDOW_HOURS), float(row.bb_upper), row.time))
        elif bool(row.short_signal):
            start = row.time + pd.Timedelta(days=1)
            setups.append(Setup(-1, start, start + pd.Timedelta(hours=RETEST_WINDOW_HOURS), float(row.bb_lower), row.time))
    setups.sort(key=lambda item: item.start)

    setup_cursor = 0
    active_setups: list[Setup] = []
    pending_entry: tuple[int, float, pd.Timestamp, pd.Timestamp] | None = None
    pending_exit_reason: str | None = None
    position: Position | None = None
    realized_total = 0.0
    trades: list[dict] = []
    equity_rows: list[dict] = []
    max_margin = 0.0
    max_notional = 0.0
    worst_adverse = 0.0

    for index, row in hourly.iterrows():
        now = row["time"]

        while setup_cursor < len(setups) and setups[setup_cursor].start <= now:
            active_setups.append(setups[setup_cursor])
            setup_cursor += 1
        active_setups = [item for item in active_setups if now <= item.expiry]

        # Execute close-generated orders at the next hourly open.
        if position is not None and pending_exit_reason is not None:
            fill = exit_fill(float(row["open"]), position.side)
            gross = (fill - position.entry_price) * position.qty * position.side
            fee = fill * position.qty * TAKER_FEE
            position.realized_parts += gross
            position.exit_fees += fee
            net = position.realized_parts - position.entry_fee - position.exit_fees
            realized_total += net
            trades.append({
                "side": "LONG" if position.side == 1 else "SHORT",
                "setup_day": position.setup_day,
                "entry_time": position.entry_time,
                "exit_time": now,
                "entry_price": position.entry_price,
                "exit_price": fill,
                "partial_taken": position.partial_done,
                "exit_reason": pending_exit_reason,
                "gross_parts": position.realized_parts,
                "fees": position.entry_fee + position.exit_fees,
                "net_pnl": net,
                "return_on_notional_pct": net / NOTIONAL * 100,
                "holding_hours": (now - position.entry_time).total_seconds() / 3600,
            })
            position = None
            pending_exit_reason = None

        if position is None and pending_entry is not None:
            side, signal_atr, setup_day, signal_time = pending_entry
            fill = entry_fill(float(row["open"]), side)
            qty = NOTIONAL / fill
            risk = STOP_ATR * signal_atr
            stop = fill - side * risk
            target = fill + side * PARTIAL_R * risk
            position = Position(
                side=side,
                entry_time=now,
                entry_price=fill,
                original_qty=qty,
                qty=qty,
                atr=signal_atr,
                stop=stop,
                target=target,
                entry_fee=NOTIONAL * TAKER_FEE,
                setup_day=setup_day,
            )
            pending_entry = None
            active_setups = []

        # Intrabar risk management. When stop and target both occur in one candle,
        # the stop is evaluated first as a conservative convention.
        if position is not None:
            stop_hit = float(row["low"]) <= position.stop if position.side == 1 else float(row["high"]) >= position.stop
            if stop_hit:
                fill = exit_fill(position.stop, position.side)
                gross = (fill - position.entry_price) * position.qty * position.side
                fee = fill * position.qty * TAKER_FEE
                position.realized_parts += gross
                position.exit_fees += fee
                net = position.realized_parts - position.entry_fee - position.exit_fees
                realized_total += net
                trades.append({
                    "side": "LONG" if position.side == 1 else "SHORT",
                    "setup_day": position.setup_day,
                    "entry_time": position.entry_time,
                    "exit_time": now,
                    "entry_price": position.entry_price,
                    "exit_price": fill,
                    "partial_taken": position.partial_done,
                    "exit_reason": "breakeven_stop" if position.partial_done else "initial_stop",
                    "gross_parts": position.realized_parts,
                    "fees": position.entry_fee + position.exit_fees,
                    "net_pnl": net,
                    "return_on_notional_pct": net / NOTIONAL * 100,
                    "holding_hours": (now - position.entry_time).total_seconds() / 3600,
                })
                position = None
                pending_exit_reason = None
            elif not position.partial_done:
                target_hit = float(row["high"]) >= position.target if position.side == 1 else float(row["low"]) <= position.target
                if target_hit:
                    close_qty = position.original_qty * 0.5
                    fill = exit_fill(position.target, position.side)
                    gross = (fill - position.entry_price) * close_qty * position.side
                    fee = fill * close_qty * TAKER_FEE
                    position.realized_parts += gross
                    position.exit_fees += fee
                    position.qty -= close_qty
                    position.partial_done = True
                    position.stop = position.entry_price

        if position is not None:
            mark_gross = (float(row["close"]) - position.entry_price) * position.qty * position.side
            unrealized = position.realized_parts + mark_gross - position.entry_fee - position.exit_fees
            adverse = (
                (float(row["low"]) / position.entry_price - 1) * 100
                if position.side == 1
                else (1 - float(row["high"]) / position.entry_price) * 100
            )
            worst_adverse = min(worst_adverse, adverse)
            notional_open = position.qty * float(row["close"])
        else:
            unrealized = 0.0
            notional_open = 0.0

        margin = notional_open / LEVERAGE
        max_margin = max(max_margin, margin)
        max_notional = max(max_notional, notional_open)
        equity_rows.append({
            "time": now,
            "realized_pnl": realized_total,
            "unrealized_pnl": unrealized,
            "equity": INITIAL_CAPITAL + realized_total + unrealized,
            "position_side": 0 if position is None else position.side,
            "open_notional": notional_open,
            "initial_margin": margin,
            "active_setups": len(active_setups),
        })

        # Decisions made only with the completed hourly candle.
        if position is not None and index + 1 < len(hourly):
            if position.side == 1:
                ema_break = pd.notna(row["ema20"]) and float(row["close"]) < float(row["ema20"])
                daily_break = pd.notna(row["prev_daily_ma20"]) and float(row["close"]) < float(row["prev_daily_ma20"])
            else:
                ema_break = pd.notna(row["ema20"]) and float(row["close"]) > float(row["ema20"])
                daily_break = pd.notna(row["prev_daily_ma20"]) and float(row["close"]) > float(row["prev_daily_ma20"])
            if daily_break:
                pending_exit_reason = "daily_ma20_break"
            elif ema_break:
                pending_exit_reason = "hourly_ema20_break"

        elif position is None and pending_entry is None and index + 1 < len(hourly):
            if pd.notna(row["atr"]) and pd.notna(row["prev_high"]) and pd.notna(row["prev_low"]):
                # Newest daily setup has priority if windows overlap.
                for setup in reversed(active_setups):
                    if setup.side == 1:
                        confirmed = (
                            float(row["low"]) <= setup.band
                            and float(row["close"]) > setup.band
                            and float(row["close"]) > float(row["open"])
                            and float(row["close"]) > float(row["prev_high"])
                        )
                    else:
                        confirmed = (
                            float(row["high"]) >= setup.band
                            and float(row["close"]) < setup.band
                            and float(row["close"]) < float(row["open"])
                            and float(row["close"]) < float(row["prev_low"])
                        )
                    if confirmed:
                        pending_entry = (setup.side, float(row["atr"]), setup.signal_day, now)
                        break

    # Force-close an open position at the final close.
    if position is not None:
        last = hourly.iloc[-1]
        fill = exit_fill(float(last["close"]), position.side)
        gross = (fill - position.entry_price) * position.qty * position.side
        fee = fill * position.qty * TAKER_FEE
        position.realized_parts += gross
        position.exit_fees += fee
        net = position.realized_parts - position.entry_fee - position.exit_fees
        realized_total += net
        trades.append({
            "side": "LONG" if position.side == 1 else "SHORT",
            "setup_day": position.setup_day,
            "entry_time": position.entry_time,
            "exit_time": last["time"],
            "entry_price": position.entry_price,
            "exit_price": fill,
            "partial_taken": position.partial_done,
            "exit_reason": "end_of_data",
            "gross_parts": position.realized_parts,
            "fees": position.entry_fee + position.exit_fees,
            "net_pnl": net,
            "return_on_notional_pct": net / NOTIONAL * 100,
            "holding_hours": (last["time"] - position.entry_time).total_seconds() / 3600,
        })

    trade_frame = pd.DataFrame(trades)
    equity_frame = pd.DataFrame(equity_rows)
    equity_frame["peak"] = equity_frame["equity"].cummax()
    equity_frame["drawdown_usdt"] = equity_frame["equity"] - equity_frame["peak"]
    equity_frame["drawdown_pct"] = equity_frame["equity"] / equity_frame["peak"] - 1

    wins = trade_frame[trade_frame["net_pnl"] > 0] if len(trade_frame) else trade_frame
    losses = trade_frame[trade_frame["net_pnl"] <= 0] if len(trade_frame) else trade_frame
    gross_profit = float(wins["net_pnl"].sum()) if len(wins) else 0.0
    gross_loss = float(-losses["net_pnl"].sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    long_trades = trade_frame[trade_frame["side"] == "LONG"] if len(trade_frame) else trade_frame
    short_trades = trade_frame[trade_frame["side"] == "SHORT"] if len(trade_frame) else trade_frame

    result = {
        "data_start": str(hourly["time"].min()),
        "data_end": str(hourly["time"].max()),
        "hourly_bars": int(len(hourly)),
        "daily_bars": int(len(daily)),
        "initial_capital_usdt": INITIAL_CAPITAL,
        "position_notional_usdt": NOTIONAL,
        "leverage": LEVERAGE,
        "daily_ma_lengths": [D_MA_FAST, D_MA_MID, D_MA_SLOW],
        "daily_bb_length": D_BB_LEN,
        "daily_bb_std": D_BB_STD,
        "daily_adx_min": D_ADX_MIN,
        "daily_max_distance_atr": D_MAX_DISTANCE_ATR,
        "retest_window_hours": RETEST_WINDOW_HOURS,
        "hourly_ema_length": H_EMA_LEN,
        "hourly_atr_length": H_ATR_LEN,
        "stop_atr": STOP_ATR,
        "partial_take_profit_r": PARTIAL_R,
        "taker_fee_each_side_pct": TAKER_FEE * 100,
        "slippage_each_side_pct": SLIPPAGE * 100,
        "funding_included": False,
        "daily_long_setups": int(daily["long_signal"].sum()),
        "daily_short_setups": int(daily["short_signal"].sum()),
        "closed_trades": int(len(trade_frame)),
        "winning_trades": int((trade_frame["net_pnl"] > 0).sum()) if len(trade_frame) else 0,
        "losing_trades": int((trade_frame["net_pnl"] <= 0).sum()) if len(trade_frame) else 0,
        "win_rate_pct": float((trade_frame["net_pnl"] > 0).mean() * 100) if len(trade_frame) else float("nan"),
        "net_profit_usdt": float(trade_frame["net_pnl"].sum()) if len(trade_frame) else 0.0,
        "final_equity_usdt": INITIAL_CAPITAL + (float(trade_frame["net_pnl"].sum()) if len(trade_frame) else 0.0),
        "profit_factor": profit_factor,
        "mdd_usdt": float(equity_frame["drawdown_usdt"].min()) if len(equity_frame) else 0.0,
        "mdd_pct": float(equity_frame["drawdown_pct"].min() * 100) if len(equity_frame) else 0.0,
        "minimum_equity_usdt": float(equity_frame["equity"].min()) if len(equity_frame) else INITIAL_CAPITAL,
        "max_open_notional_usdt": max_notional,
        "max_initial_margin_usdt": max_margin,
        "worst_intraday_adverse_move_pct": worst_adverse,
        "long_trades": int(len(long_trades)),
        "long_win_rate_pct": float((long_trades["net_pnl"] > 0).mean() * 100) if len(long_trades) else float("nan"),
        "short_trades": int(len(short_trades)),
        "short_win_rate_pct": float((short_trades["net_pnl"] > 0).mean() * 100) if len(short_trades) else float("nan"),
        "partial_tp_trades": int(trade_frame["partial_taken"].sum()) if len(trade_frame) else 0,
        "best_trade_usdt": float(trade_frame["net_pnl"].max()) if len(trade_frame) else float("nan"),
        "worst_trade_usdt": float(trade_frame["net_pnl"].min()) if len(trade_frame) else float("nan"),
        "average_trade_usdt": float(trade_frame["net_pnl"].mean()) if len(trade_frame) else float("nan"),
        "median_trade_usdt": float(trade_frame["net_pnl"].median()) if len(trade_frame) else float("nan"),
        "average_holding_hours": float(trade_frame["holding_hours"].mean()) if len(trade_frame) else float("nan"),
    }
    result["suggested_cash_floor_usdt"] = max_margin + max(0.0, -result["mdd_usdt"]) + 50.0

    trade_frame.to_csv(OUT / "trades.csv", index=False)
    equity_frame.to_csv(OUT / "equity.csv", index=False)
    with open(OUT / "summary.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=True)

    return trade_frame, equity_frame, result


def write_report(result: dict) -> None:
    text = f"""# BTCUSDT Daily/1H Hybrid Backtest

- Data: {result['data_start']} to {result['data_end']}
- Daily direction: BB220, MA20/60/200, ADX >= 25
- Entry: 1H retest and reversal confirmation within 72 hours
- Exit: 1.5 ATR stop, 50% at 1R, then 1H EMA20 or previous completed daily MA20 break
- Position: 100 USDT, leverage 10x
- Fee/slippage: {result['taker_fee_each_side_pct']:.3f}% / {result['slippage_each_side_pct']:.3f}% each side
- Funding: excluded

## Results

- Daily setups: {result['daily_long_setups']} long / {result['daily_short_setups']} short
- Closed trades: {result['closed_trades']}
- Win rate: {result['win_rate_pct']:.2f}%
- Net profit: {result['net_profit_usdt']:.2f} USDT
- Profit factor: {result['profit_factor']:.3f}
- MDD: {result['mdd_usdt']:.2f} USDT ({result['mdd_pct']:.2f}%)
- Long win rate: {result['long_win_rate_pct']:.2f}% ({result['long_trades']} trades)
- Short win rate: {result['short_win_rate_pct']:.2f}% ({result['short_trades']} trades)
- Partial TP trades: {result['partial_tp_trades']}
- Best / worst trade: {result['best_trade_usdt']:.2f} / {result['worst_trade_usdt']:.2f} USDT
- Average holding: {result['average_holding_hours']:.2f} hours
- Suggested cash floor: {result['suggested_cash_floor_usdt']:.2f} USDT
"""
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    hourly_raw = download_hourly()
    hourly, daily = build_indicators(hourly_raw)
    _, _, result = run_backtest(hourly, daily)
    write_report(result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
