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


# --------------------------- 获取K线 --------------------------- #
def _get_kline_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=start,
            end_date=end,
            freq="D",
            api=pro
        )
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume", "amount"]
    ].copy()

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


# --------------------------- 获取股本 --------------------------- #
def _get_capital_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = pro.daily_basic(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="trade_date,total_share"
        )
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "total_share": "capital"})
    df["date"] = pd.to_datetime(df["date"])
    df["capital"] = pd.to_numeric(df["capital"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


# --------------------------- 过滤 --------------------------- #
def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    code = df["symbol"].astype(str).str.zfill(6)
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~code.str.startswith(("300", "301"))
    if "star" in exclude_boards:
        mask &= ~code.str.startswith(("688",))
    if "bj" in exclude_boards:
        mask &= ~code.str.startswith(("4", "8"))
    return df[mask].copy()


def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).unique().tolist()
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes


# --------------------------- 单股抓取（核心增量逻辑） --------------------------- #
def fetch_one(code: str, default_start: str, end_date: str, out_dir: Path):
    csv_path = out_dir / f"{code}.csv"
    actual_start = default_start

    # 1. 增量检查逻辑
    if csv_path.exists():
        try:
            # 只读最后一行日期，极大提升检查速度
            last_line = pd.read_csv(csv_path, usecols=['date']).tail(1)
            if not last_line.empty:
                last_date_str = str(last_line['date'].iloc[-1]).replace('-', '').split(' ')[0]
                last_date_dt = dt.datetime.strptime(last_date_str, '%Y%m%d')
                # 真正需要同步的是下一天
                actual_start = (last_date_dt + dt.timedelta(days=1)).strftime('%Y%m%d')
        except Exception as e:
            logger.debug(f"{code} 本地文件解析失败，将从默认日期全量同步: {e}")

    # 2. 如果已最新，跳过
    if actual_start > end_date:
        return

    # 3. 抓取与追加
    for attempt in range(1, 4):
        try:
            k_df = _get_kline_tushare(code, actual_start, end_date)
            if k_df.empty:
                return  # 无新行情需要追加

            cap_df = _get_capital_tushare(code, actual_start, end_date)

            if not cap_df.empty:
                df = pd.merge(k_df, cap_df, on="date", how="left")
                df["capital"] = df["capital"].ffill()
            else:
                df = k_df.copy()
                df["capital"] = None

            # 格式化日期为字符串，方便后续 CSV 读取
            df["date"] = df["date"].dt.strftime('%Y-%m-%d')

            # 4. 追加写入 (mode='a')
            is_new = not csv_path.exists()
            # 如果是新文件，写 header；如果是追加，不写 header
            df.to_csv(csv_path, mode='a', index=False, header=is_new, encoding='utf-8')
            break

        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 5 * attempt
                time.sleep(silent_seconds)
    else:
        logger.error("%s 更新失败！", code)


# --------------------------- 主入口 --------------------------- #
def main():
    # main参数处理
    parser = argparse.ArgumentParser(description="Tushare 增量下载器")
    parser.add_argument("--start", default="20250101", help="起始日期")
    parser.add_argument("--end", default="today", help="结束日期")
    parser.add_argument("--stocklist", type=Path, default=Path("./stocklist.csv"))
    parser.add_argument("--exclude-boards", nargs="*", default=[], choices=["gem", "star", "bj"])
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=6, help="并发数")
    args = parser.parse_args()

    # Tushare 初始化
    ts_token = "26d6a5877ad3da85312145f9975b98873a5291b5a50e3af33d6b77671b70"
    global pro
    pro = ts.pro_api(ts_token)
    pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'

    # 日期参数逻辑处理，若在3点前运行且end是today，则同步昨天数据
    now = dt.datetime.now()
    if str(args.end).lower() == "today":
        # 15:30分前运行，数据通常未结算，同步到昨天
        if now.hour < 15 or (now.hour == 15 and now.minute < 30):
            target_end = (now - dt.timedelta(days=1)).strftime("%Y%m%d")
        else:
            target_end = now.strftime("%Y%m%d")
    else:
        target_end = args.end

    # 定义输出目录
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_boards = set(args.exclude_boards or [])
    codes = load_codes_from_stocklist(args.stocklist, exclude_boards)

    # 命令行打印，并发起多线程抓取
    logger.info(f"任务启动: 从 {args.start} 到 {target_end}, 线程数: {args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_one, code, args.start, target_end, out_dir)
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="同步进度"):
            pass

    logger.info("全部完成！")


if __name__ == "__main__":
    main()
