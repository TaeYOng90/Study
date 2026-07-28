from __future__ import annotations

import csv
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
INTERVAL = "1d"
START_MONTH = date(2019, 9, 1)
TODAY = datetime.now(timezone.utc).date()
LAST_COMPLETE_DAY = TODAY - timedelta(days=1)
NOTIONAL_PER_ENTRY = 100.0
LEVERAGE = 10.0
INITIAL_CAPITAL = 1000.0
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002
MA_LEN = 20
BB_LEN = 220
BB_STD = 2.0
OUT = Path("tmp_backtest/results")
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
            r = requests.get(url, timeout=60, headers={"User-Agent": "btc-backtest/1.0"})
            if r.status_code == 404:
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            last = exc
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise last


def parse_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("ZIP contains no CSV")
        raw = zf.read(names[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if len(df.columns) != 12:
        df = pd.read_csv(io.BytesIO(raw))
        df = df.iloc[:, :12]
        df.columns = COLUMNS
    else:
        # Some newer archives include a header row.
        first = str(df.iloc[0, 0]).lower()
        if "open" in first:
            df = df.iloc[1:].reset_index(drop=True)
        df.columns = COLUMNS
    return df


def download_data() -> pd.DataFrame:
    frames = []
    last_month_start = date(LAST_COMPLETE_DAY.year, LAST_COMPLETE_DAY.month, 1)
    # Complete prior months from monthly archive.
    prior_month_end = last_month_start - timedelta(days=1)
    for m in month_iter(START_MONTH, date(prior_month_end.year, prior_month_end.month, 1)):
        ym = m.strftime("%Y-%m")
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ym}.zip"
        try:
            frames.append(parse_zip(get_bytes(url)))
        except FileNotFoundError:
            print("missing monthly", url)

    # Current month complete daily files.
    d = last_month_start
    while d <= LAST_COMPLETE_DAY:
        ds = d.isoformat()
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{ds}.zip"
        try:
            frames.append(parse_zip(get_bytes(url)))
        except FileNotFoundError:
            print("missing daily", url)
        d += timedelta(days=1)

    if not frames:
        raise RuntimeError("No data downloaded")
    df = pd.concat(frames, ignore_index=True)
    for c in ["open_time", "close_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df = df[df["date"].dt.date <= LAST_COMPLETE_DAY].reset_index(drop=True)
    df.to_csv(OUT / "BTCUSDT_1d.csv", index=False)
    return df


@dataclass
class Lot:
    entry_date: pd.Timestamp
    entry_price: float
    qty: float
    entry_fee: float


def backtest(df: pd.DataFrame):
    df = df.copy()
    df["ma20"] = df["close"].rolling(MA_LEN).mean()
    df["bb_mid"] = df["close"].rolling(BB_LEN).mean()
    std = df["close"].rolling(BB_LEN).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + BB_STD * std
    df["bb_lower"] = df["bb_mid"] - BB_STD * std
    df["long_signal"] = (df["close"] > df["ma20"]) & (df["close"] > df["bb_upper"])
    df["short_signal"] = (df["close"] < df["ma20"]) & (df["close"] < df["bb_lower"])

    side = 0
    lots: list[Lot] = []
    pending_exit = False
    pending_entry = 0
    realized = 0.0
    equity_rows = []
    cycles = []
    cycle_start = None
    cycle_entries = 0
    max_notional = 0.0
    max_margin = 0.0
    peak_adverse_pct = 0.0

    for i in range(BB_LEN, len(df)):
        row = df.iloc[i]
        px_open = float(row.open)

        # Execute yesterday's decision at today's open. Exit has priority.
        if pending_exit and side != 0:
            exec_px = px_open * (1 - SLIPPAGE if side == 1 else 1 + SLIPPAGE)
            gross = sum((exec_px - lot.entry_price) * lot.qty * side for lot in lots)
            exit_notional = sum(lot.qty for lot in lots) * exec_px
            exit_fee = exit_notional * TAKER_FEE
            total_entry_fee = sum(l.entry_fee for l in lots)
            pnl = gross - exit_fee - total_entry_fee
            realized += pnl
            cycles.append({
                "side": "LONG" if side == 1 else "SHORT",
                "entry_date": cycle_start.isoformat(),
                "exit_date": row.date.isoformat(),
                "entries": cycle_entries,
                "entry_notional": cycle_entries * NOTIONAL_PER_ENTRY,
                "exit_price": exec_px,
                "gross_pnl": gross,
                "fees": total_entry_fee + exit_fee,
                "net_pnl": pnl,
                "return_on_total_notional_pct": pnl / (cycle_entries * NOTIONAL_PER_ENTRY) * 100,
            })
            side = 0
            lots = []
            pending_exit = False
            pending_entry = 0
            cycle_start = None
            cycle_entries = 0
        elif pending_entry != 0:
            if side == 0:
                side = pending_entry
                cycle_start = row.date
            if side == pending_entry:
                exec_px = px_open * (1 + SLIPPAGE if side == 1 else 1 - SLIPPAGE)
                qty = NOTIONAL_PER_ENTRY / exec_px
                fee = NOTIONAL_PER_ENTRY * TAKER_FEE
                lots.append(Lot(row.date, exec_px, qty, fee))
                cycle_entries += 1
            pending_entry = 0

        # Intraday mark-to-market and risk.
        if side != 0 and lots:
            mark = float(row.close)
            unreal = sum((mark - lot.entry_price) * lot.qty * side for lot in lots) - sum(l.entry_fee for l in lots)
            avg_entry = sum(l.entry_price * l.qty for l in lots) / sum(l.qty for l in lots)
            adverse = ((float(row.low) / avg_entry - 1) if side == 1 else (1 - float(row.high) / avg_entry)) * 100
            peak_adverse_pct = min(peak_adverse_pct, adverse)
        else:
            unreal = 0.0

        notional = len(lots) * NOTIONAL_PER_ENTRY
        margin = notional / LEVERAGE
        max_notional = max(max_notional, notional)
        max_margin = max(max_margin, margin)
        equity_rows.append({
            "date": row.date,
            "realized_pnl": realized,
            "unrealized_pnl": unreal,
            "equity": INITIAL_CAPITAL + realized + unreal,
            "position_side": side,
            "entries_open": len(lots),
            "position_notional": notional,
            "initial_margin": margin,
        })

        # Decide at close for next day's open.
        if i + 1 < len(df):
            if side == 1 and float(row.low) <= float(row.ma20):
                pending_exit = True
            elif side == -1 and float(row.high) >= float(row.ma20):
                pending_exit = True
            elif side == 0:
                if bool(row.long_signal):
                    pending_entry = 1
                elif bool(row.short_signal):
                    pending_entry = -1
            elif side == 1 and bool(row.long_signal):
                pending_entry = 1
            elif side == -1 and bool(row.short_signal):
                pending_entry = -1

    eq = pd.DataFrame(equity_rows)
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown_usdt"] = eq["equity"] - eq["peak"]
    eq["drawdown_pct"] = eq["equity"] / eq["peak"] - 1
    trades = pd.DataFrame(cycles)

    if len(trades):
        wins = trades[trades.net_pnl > 0]
        losses = trades[trades.net_pnl <= 0]
        gp = wins.net_pnl.sum()
        gl = -losses.net_pnl.sum()
        pf = gp / gl if gl > 0 else math.inf
        win_rate = len(wins) / len(trades) * 100
    else:
        wins = losses = trades
        pf = float("nan")
        win_rate = float("nan")

    min_equity = float(eq.equity.min()) if len(eq) else INITIAL_CAPITAL
    result = {
        "data_start": df.date.min().isoformat(),
        "data_end": df.date.max().isoformat(),
        "bars": int(len(df)),
        "strategy_start": df.iloc[BB_LEN].date.isoformat(),
        "initial_capital_usdt": INITIAL_CAPITAL,
        "notional_per_entry_usdt": NOTIONAL_PER_ENTRY,
        "leverage": LEVERAGE,
        "ma_length": MA_LEN,
        "bb_length": BB_LEN,
        "bb_std": BB_STD,
        "taker_fee_each_side_pct": TAKER_FEE * 100,
        "slippage_each_side_pct": SLIPPAGE * 100,
        "funding_included": False,
        "closed_cycles": int(len(trades)),
        "winning_cycles": int((trades.net_pnl > 0).sum()) if len(trades) else 0,
        "losing_cycles": int((trades.net_pnl <= 0).sum()) if len(trades) else 0,
        "win_rate_pct": win_rate,
        "net_profit_usdt": float(trades.net_pnl.sum()) if len(trades) else 0.0,
        "profit_factor": pf,
        "mdd_usdt": float(eq.drawdown_usdt.min()) if len(eq) else 0.0,
        "mdd_pct": float(eq.drawdown_pct.min() * 100) if len(eq) else 0.0,
        "minimum_equity_usdt": min_equity,
        "max_position_notional_usdt": max_notional,
        "max_initial_margin_usdt": max_margin,
        "max_open_entries": int(eq.entries_open.max()) if len(eq) else 0,
        "worst_intraday_adverse_move_from_weighted_entry_pct": peak_adverse_pct,
        "rough_minimum_starting_cash_usdt": max_margin + max(0.0, -float(eq.drawdown_usdt.min())) + 20.0 if len(eq) else 20.0,
    }
    if len(trades):
        for name, val in {
            "long_cycles": int((trades.side == "LONG").sum()),
            "short_cycles": int((trades.side == "SHORT").sum()),
            "long_win_rate_pct": float((trades.loc[trades.side == "LONG", "net_pnl"] > 0).mean() * 100) if (trades.side == "LONG").any() else float("nan"),
            "short_win_rate_pct": float((trades.loc[trades.side == "SHORT", "net_pnl"] > 0).mean() * 100) if (trades.side == "SHORT").any() else float("nan"),
            "best_cycle_usdt": float(trades.net_pnl.max()),
            "worst_cycle_usdt": float(trades.net_pnl.min()),
        }.items():
            result[name] = val

    return df, trades, eq, result


def write_report(result: dict):
    def f(x, n=2):
        return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{n}f}"
    lines = [
        "# BTCUSDT 일봉 전략 백테스트",
        "",
        f"- 데이터: {result['data_start'][:10]} ~ {result['data_end'][:10]} ({result['bars']:,}개 일봉)",
        f"- 실제 전략 계산 시작: {result['strategy_start'][:10]} (BB220 워밍업 이후)",
        "- 체결: 신호 확인 다음 일봉 시가, 시장가",
        f"- 비용: 진입·청산 각각 수수료 {result['taker_fee_each_side_pct']:.3f}% + 슬리피지 {result['slippage_each_side_pct']:.3f}%",
        "- 펀딩비: 제외",
        "",
        "## 결과",
        "",
        f"- 종료된 포지션 사이클: **{result['closed_cycles']}회**",
        f"- 승률: **{f(result['win_rate_pct'])}%**",
        f"- 순손익: **{f(result['net_profit_usdt'])} USDT**",
        f"- Profit Factor: **{f(result['profit_factor'])}**",
        f"- MDD: **{f(result['mdd_usdt'])} USDT ({f(result['mdd_pct'])}%)** — 초기자본 1,000 USDT 기준, 일별 평가손익 포함",
        f"- 최대 누적 포지션: **{f(result['max_position_notional_usdt'])} USDT**",
        f"- 최대 10배 초기증거금: **{f(result['max_initial_margin_usdt'])} USDT**",
        f"- 최대 추가진입 횟수: **{result['max_open_entries']}회**",
        f"- 보수적 최소 준비자금 추정: **{f(result['rough_minimum_starting_cash_usdt'])} USDT**",
        "",
        "## 주의",
        "",
        "- 100 USDT는 증거금이 아니라 포지션 명목가치이므로, 10배 레버리지는 손익을 10배로 만들지 않습니다. 필요한 초기증거금만 약 1/10로 줄입니다.",
        "- 정확한 강제청산은 유지증거금 구간, 누적 평균단가, 교차/격리 설정에 따라 달라져 이 결과에 반영하지 않았습니다.",
        "- 하루 한 번 조회 조건에 맞춰 MA20 장중 터치를 당일 즉시 체결하지 않고 다음 날 시가에 청산했습니다.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df = download_data()
    indicator_df, trades, eq, result = backtest(df)
    indicator_df.to_csv(OUT / "daily_with_indicators.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    eq.to_csv(OUT / "equity_curve.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_report(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
