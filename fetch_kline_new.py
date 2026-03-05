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

# --------------------------- 全局配置 --------------------------- #
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


# --------------------------- 工具函数 --------------------------- #
def _to_ts_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in ["访问频繁", "请稍后", "超过频率", "too many requests", "429", "403"])


# --------------------------- 数据抓取与适配 --------------------------- #
def _get_kline_tushare(code: str, start: str, end: str, pro_api) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = ts.pro_bar(ts_code=ts_code, adj="qfq", start_date=start, end_date=end, freq="D", api=pro_api)
        if df is None or df.empty: return pd.DataFrame()

        # 兼容性修复：适配 trade_date 或 date
        if 'trade_date' in df.columns:
            df = df.rename(columns={"trade_date": "date"})
        elif 'date' not in df.columns:
            return pd.DataFrame()

        if 'vol' in df.columns:
            df = df.rename(columns={"vol": "volume"})

        target_cols = ["date", "open", "close", "high", "low", "volume", "amount"]
        df = df[target_cols].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()


def _get_capital_tushare(code: str, start: str, end: str, pro_api) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = pro_api.daily_basic(ts_code=ts_code, start_date=start, end_date=end, fields="trade_date,total_share")
        if df is None or df.empty: return pd.DataFrame()

        date_col = 'trade_date' if 'trade_date' in df.columns else 'date'
        df = df.rename(columns={date_col: "date", "total_share": "capital"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "capital"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# --------------------------- 核心逻辑 --------------------------- #
def fetch_one(code: str, default_start: str, end_date: str, out_dir: Path, pro_api):
    csv_path = out_dir / f"{code}.csv"
    actual_start = default_start

    if csv_path.exists():
        try:
            # 增量检查：读取最后一行
            last_df = pd.read_csv(csv_path, usecols=['date']).tail(1)
            if not last_df.empty:
                last_date_str = str(last_df['date'].iloc[-1]).replace('-', '').replace('/', '')[:8]
                if last_date_str >= end_date: return
                last_dt = dt.datetime.strptime(last_date_str, '%Y%m%d')
                actual_start = (last_dt + dt.timedelta(days=1)).strftime('%Y%m%d')
        except Exception:
            pass

    if actual_start > end_date: return

    k_df = _get_kline_tushare(code, actual_start, end_date, pro_api)
    if k_df.empty: return

    cap_df = _get_capital_tushare(code, actual_start, end_date, pro_api)
    df = pd.merge(k_df, cap_df, on="date", how="left") if not cap_df.empty else k_df.assign(capital=None)

    # 格式化：强制 4 位小数并转为字符串写入，确保格式统一
    df["capital"] = df["capital"].ffill()
    df["date"] = df["date"].dt.strftime('%Y-%m-%d')

    num_cols = ["open", "close", "high", "low", "volume", "amount", "capital"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').map(lambda x: f"{x:.4f}" if pd.notnull(x) else "0.0000")

    is_new = not csv_path.exists()
    df.to_csv(csv_path, mode='a', index=False, header=is_new, encoding='utf-8')


# --------------------------- 主程序 --------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Tushare 增量下载器")
    parser.add_argument("--start", default="20250101")
    parser.add_argument("--end", default="today")
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"))
    # 把这个参数加回来了！
    parser.add_argument("--exclude-boards", nargs="*", default=[], choices=["gem", "star", "bj"])
    parser.add_argument("--out", default="./data")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # 初始化 Tushare
    TOKEN = "26d6a5877ad3da85312145f9975b98873a5291b5a50e3af33d6b77671b70"
    ts.set_token(TOKEN)
    pro = ts.pro_api(TOKEN)
    pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'

    # 日期判定
    now = dt.datetime.now()
    target_end = now.strftime("%Y%m%d") if args.end == "today" else args.end

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载并过滤股票
    if not args.stocklist.exists():
        logger.error(f"找不到列表文件: {args.stocklist}")
        return

    df_list = pd.read_csv(args.stocklist)
    col = "symbol" if "symbol" in df_list.columns else "code"
    df_list[col] = df_list[col].astype(str).str.zfill(6)

    # 过滤板块逻辑
    exclude = set(args.exclude_boards)
    if "gem" in exclude:
        df_list = df_list[~df_list[col].str.startswith(("300", "301"))]
    if "star" in exclude:
        df_list = df_list[~df_list[col].str.startswith("688")]
    if "bj" in exclude:
        df_list = df_list[~df_list[col].str.startswith(("4", "8"))]

    codes = df_list[col].unique().tolist()
    logger.info(f"任务启动 | 目标日期: {target_end} | 股票总数: {len(codes)} (已排除 {args.exclude_boards})")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_one, code, args.start, target_end, out_dir, pro) for code in codes]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="数据同步进度"):
            pass

    logger.info("全部同步完成！")


if __name__ == "__main__":
    main()