from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tushare as ts
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_incremental")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded"
)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    pass

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

# --------------------------- Tushare 会话 --------------------------- #
pro: Optional[ts.pro_api] = None

def _to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"

# --------------------------- 数据获取 --------------------------- #
def _get_kline_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = ts.pro_bar(
            ts_code=ts_code, adj="qfq", start_date=start, end_date=end, freq="D", api=pro
        )
    except Exception as e:
        if _looks_like_ip_ban(e): raise RateLimitError(str(e)) from e
        raise
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume", "amount"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def _get_capital_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = pro.daily_basic(ts_code=ts_code, start_date=start, end_date=end, fields="trade_date,total_share")
    except Exception as e:
        if _looks_like_ip_ban(e): raise RateLimitError(str(e)) from e
        raise
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns={"trade_date": "date", "total_share": "capital"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

# --------------------------- 过滤逻辑（还原） --------------------------- #
def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    # 兼容 symbol 或 code 列名
    col = "symbol" if "symbol" in df.columns else "code"
    code_series = df[col].astype(str).str.zfill(6)
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~code_series.str.startswith(("300", "301"))
    if "star" in exclude_boards:
        mask &= ~code_series.str.startswith(("688",))
    if "bj" in exclude_boards:
        mask &= ~code_series.str.startswith(("4", "8"))
    return df[mask].copy()

def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)
    df = _filter_by_boards_stocklist(df, exclude_boards)
    col = "symbol" if "symbol" in df.columns else "code"
    codes = df[col].astype(str).str.zfill(6).unique().tolist()
    logger.info("从 %s 读取到 %d 只股票 (排除板块: %s)",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

# --------------------------- 单股抓取 --------------------------- #
def fetch_one(code: str, default_start: str, end_date: str, out_dir: Path):
    csv_path = out_dir / f"{code}.csv"
    actual_start = default_start

    if csv_path.exists():
        try:
            last_line = pd.read_csv(csv_path, usecols=['date']).tail(1)
            if not last_line.empty:
                last_date_str = str(last_line['date'].iloc[-1]).split(' ')[0].replace('-', '')
                if last_date_str >= end_date:
                    return
                last_date_dt = dt.datetime.strptime(last_date_str, '%Y%m%d')
                actual_start = (last_date_dt + dt.timedelta(days=1)).strftime('%Y%m%d')
        except Exception as e:
            logger.debug(f"{code} 解析失败: {e}")

    if actual_start > end_date: return

    for attempt in range(1, 4):
        try:
            k_df = _get_kline_tushare(code, actual_start, end_date)
            if k_df.empty: return
            cap_df = _get_capital_tushare(code, actual_start, end_date)
            df = pd.merge(k_df, cap_df, on="date", how="left") if not cap_df.empty else k_df.assign(capital=None)
            df["capital"] = df["capital"].ffill()
            df["date"] = df["date"].dt.strftime('%Y-%m-%d')
            is_new = not csv_path.exists()
            df.to_csv(csv_path, mode='a', index=False, header=is_new, encoding='utf-8')
            break
        except Exception as e:
            if _looks_like_ip_ban(e): _cool_sleep(COOLDOWN_SECS)
            else: time.sleep(2 * attempt)

# --------------------------- 主入口 --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Tushare 增量下载器")
    parser.add_argument("--start", default="20250101", help="起始日期")
    parser.add_argument("--end", default="today", help="结束日期")
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"))
    parser.add_argument("--exclude-boards", nargs="*", default=[], choices=["gem", "star", "bj"])
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=6, help="线程数")
    args = parser.parse_args()

    # Tushare 初始化
    ts_token = "26d6a5877ad3da85312145f9975b98873a5291b5a50e3af33d6b77671b70"
    global pro
    pro = ts.pro_api(ts_token)
    pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'

    # 1. 判定最新可用交易日 (跳过逻辑核心)
    now = dt.datetime.now()
    try:
        cal_df = pro.trade_cal(is_open=1, start_date=(now - dt.timedelta(days=5)).strftime('%Y%m%d'), end_date=now.strftime('%Y%m%d'))
        last_trade_day = cal_df.iloc[-1]['cal_date']
        # 15:45 前认为最新完整数据是前一交易日
        if now.strftime("%Y%m%d") == last_trade_day and now.time() < dt.time(15, 45):
            effective_end = cal_df.iloc[-2]['cal_date']
        else:
            effective_end = last_trade_day
    except Exception:
        effective_end = now.strftime("%Y%m%d")

    target_end = effective_end if args.end == "today" else args.end

    # 2. 样本预检 (跳过全量扫描)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_files = list(out_dir.glob("*.csv"))
    if sample_files:
        sample_path = random.choice(sample_files)
        try:
            sample_date = str(pd.read_csv(sample_path, usecols=['date']).tail(1)['date'].iloc[-1]).replace('-', '')
            if sample_date >= target_end:
                logger.info(f"【智能跳过】样本 {sample_path.name} 已是最新({sample_date})，无需更新。")
                return
        except: pass

    # 3. 加载并过滤代码
    exclude_boards = set(args.exclude_boards or [])
    codes = load_codes_from_stocklist(args.stocklist, exclude_boards)

    logger.info(f"任务启动: 结束日期 {target_end}, 线程数 {args.workers}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_one, code, args.start, target_end, out_dir) for code in codes]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="同步进度"): pass

    logger.info("全部完成！")

if __name__ == "__main__":
    main()