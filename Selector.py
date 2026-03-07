from typing import Dict, List, Optional, Any
from scipy.signal import find_peaks
import logging
import os
from datetime import datetime
from typing import List, Any, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger("selector")


# --------------------------- 通用指标 --------------------------- #

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100

    K = np.zeros_like(rsv, dtype=float)
    D = np.zeros_like(rsv, dtype=float)
    for i in range(len(df)):
        if i == 0:
            K[i] = D[i] = 50.0
        else:
            K[i] = 2 / 3 * K[i - 1] + 1 / 3 * rsv.iloc[i]
            D[i] = 2 / 3 * D[i - 1] + 1 / 3 * K[i]
    J = 3 * K - 2 * D
    return df.assign(K=K, D=D, J=J)


def compute_bbi(df: pd.DataFrame) -> pd.Series:
    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def compute_rsv(
        df: pd.DataFrame,
        n: int,
) -> pd.Series:
    """
    按公式：RSV(N) = 100 × (C - LLV(L,N)) ÷ (HHV(C,N) - LLV(L,N))
    - C 用收盘价最高值 (HHV of close)
    - L 用最低价最低值 (LLV of low)
    """
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_close_n = df["close"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_close_n - low_n + 1e-9) * 100.0
    return rsv


def compute_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """计算 MACD 指标中的 DIF (EMA fast - EMA slow)。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def bbi_deriv_uptrend(
        bbi: pd.Series,
        *,
        min_window: int,
        max_window: int or None = None,
        q_threshold: float = 0.0,
) -> bool:
    """
    判断 BBI 是否“整体上升”。

    令最新交易日为 T，在区间 [T-w+1, T]（w 自适应，w ≥ min_window 且 ≤ max_window）
    内，先将 BBI 归一化：BBI_norm(t) = BBI(t) / BBI(T-w+1)。

    再计算一阶差分 Δ(t) = BBI_norm(t) - BBI_norm(t-1)。  
    若 Δ(t) 的前 q_threshold 分位数 ≥ 0，则认为该窗口通过；只要存在
    **最长** 满足条件的窗口即可返回 True。q_threshold=0 时退化为
    “全程单调不降”（旧版行为）。

    Parameters
    ----------
    bbi : pd.Series
        BBI 序列（最新值在最后一位）。
    min_window : int
        检测窗口的最小长度。
    max_window : int or None
        检测窗口的最大长度；None 表示不设上限。
    q_threshold : float, default 0.0
        允许一阶差分为负的比例（0 ≤ q_threshold ≤ 1）。
    """
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold 必须位于 [0, 1] 区间内")

    bbi = bbi.dropna()
    if len(bbi) < min_window:
        return False

    longest = min(len(bbi), max_window or len(bbi))

    # 自最长窗口向下搜索，找到任一满足条件的区间即通过
    for w in range(longest, min_window - 1, -1):
        seg = bbi.iloc[-w:]  # 区间 [T-w+1, T]
        norm = seg / seg.iloc[0]  # 归一化
        diffs = np.diff(norm.values)  # 一阶差分
        if np.quantile(diffs, q_threshold) >= 0:
            return True
    return False


def _find_peaks(
        df: pd.DataFrame,
        *,
        column: str = "high",
        distance: Optional[int] = None,
        prominence: Optional[float] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
        rel_height: float = 0.5,
        **kwargs: Any,
) -> pd.DataFrame:
    if column not in df.columns:
        raise KeyError(f"'{column}' not found in DataFrame columns: {list(df.columns)}")

    y = df[column].to_numpy()

    indices, props = find_peaks(
        y,
        distance=distance,
        prominence=prominence,
        height=height,
        width=width,
        rel_height=rel_height,
        **kwargs,
    )

    peaks_df = df.iloc[indices].copy()
    peaks_df["is_peak"] = True

    # Flatten SciPy arrays into columns (only those with same length as indices)
    for key, arr in props.items():
        if isinstance(arr, (list, np.ndarray)) and len(arr) == len(indices):
            peaks_df[f"peak_{key}"] = arr

    return peaks_df


def last_valid_ma_cross_up(
        close: pd.Series,
        ma: pd.Series,
        lookback_n: int or None = None,
) -> Optional[int]:
    """
    查找“有效上穿 MA”的最后一个交易日 T（close[T-1] < ma[T-1] 且 close[T] ≥ ma[T]）。
    - 返回的是 **整数位置**（iloc 用）。
    - lookback_n: 仅在最近 N 根内查找；None 则全历史。
    """
    n = len(close)
    start = 1  # 至少要从 1 起，因为要看 T-1
    if lookback_n is not None:
        start = max(start, n - lookback_n)

    # 自后向前找最后一次有效上穿
    for i in range(n - 1, start - 1, -1):
        if i - 1 < 0:
            continue
        c_prev, c_now = close.iloc[i - 1], close.iloc[i]
        m_prev, m_now = ma.iloc[i - 1], ma.iloc[i]
        if pd.notna(c_prev) and pd.notna(c_now) and pd.notna(m_prev) and pd.notna(m_now):
            if c_prev < m_prev and c_now >= m_now:
                return i
    return None


def compute_zx_lines(
        df: pd.DataFrame,
        m1: int = 14, m2: int = 28, m3: int = 57, m4: int = 114
) -> tuple[pd.Series, pd.Series]:
    """返回 (ZXDQ, ZXDKX)
    ZXDQ = EMA(EMA(C,10),10)
    ZXDKX = (MA(C,14)+MA(C,28)+MA(C,57)+MA(C,114))/4
    """
    close = df["close"].astype(float)
    zxdq = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()

    ma1 = close.rolling(window=m1, min_periods=m1).mean()
    ma2 = close.rolling(window=m2, min_periods=m2).mean()
    ma3 = close.rolling(window=m3, min_periods=m3).mean()
    ma4 = close.rolling(window=m4, min_periods=m4).mean()
    zxdkx = (ma1 + ma2 + ma3 + ma4) / 4.0
    return zxdq, zxdkx


def passes_day_constraints_today(df: pd.DataFrame, pct_limit: float = 0.02, amp_limit: float = 0.07) -> bool:
    """
    所有战法的统一当日过滤：
    1) 当前交易日相较于前一日涨跌幅 < pct_limit（绝对值）
    2) 当日振幅（High-Low 相对 Low） < amp_limit
    """
    if len(df) < 2:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close_today = float(last["close"])
    close_yest = float(prev["close"])
    high_today = float(last["high"])
    low_today = float(last["low"])
    if close_yest <= 0 or low_today <= 0:
        return False
    pct_chg = abs(close_today / close_yest - 1.0)
    amplitude = (high_today - low_today) / low_today
    return (pct_chg < pct_limit) and (amplitude < amp_limit)


def zx_condition_at_positions(
        df: pd.DataFrame,
        *,
        require_close_gt_long: bool = True,
        require_short_gt_long: bool = True,
        pos: int or None = None,
) -> bool:
    """
    在指定位置 pos（iloc 位置；None 表示当日）检查知行条件：
      - 收盘 > 长期线（可选）
      - 短期线 > 长期线（可选）
    注：长期线需满样本；若为 NaN 直接返回 False。
    """
    if df.empty:
        return False
    zxdq, zxdkx = compute_zx_lines(df)
    if pos is None:
        pos = len(df) - 1

    if pos < 0 or pos >= len(df):
        return False

    s = float(zxdq.iloc[pos])
    l = float(zxdkx.iloc[pos]) if pd.notna(zxdkx.iloc[pos]) else float("nan")
    c = float(df["close"].iloc[pos])

    if not np.isfinite(l) or not np.isfinite(s):
        return False

    if require_close_gt_long and not (c > l):
        return False
    if require_short_gt_long and not (s > l):
        return False
    return True


# --------------------------- Selector 类 --------------------------- #
class BBIKDJSelector:
    """
    自适应 *BBI(导数)* + *KDJ* 选股器
        • BBI: 允许 bbi_q_threshold 比例的回撤
        • KDJ: J < threshold ；或位于历史 J 的 j_q_threshold 分位及以下
        • MACD: DIF > 0
        • 收盘价波动幅度 ≤ price_range_pct
    """

    def __init__(
            self,
            j_threshold: float = -5,
            bbi_min_window: int = 90,
            max_window: int = 90,
            price_range_pct: float = 100.0,
            bbi_q_threshold: float = 0.05,
            j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.bbi_q_threshold = bbi_q_threshold  # ← 原 q_threshold
        self.j_q_threshold = j_q_threshold  # ← 新增

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        if not passes_day_constraints_today(hist):
            return False

        # 0. 收盘价波动幅度约束（最近 max_window 根 K 线）
        win = hist.tail(self.max_window)
        high, low = win["close"].max(), win["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
                hist["BBI"],
                min_window=self.bbi_min_window,
                max_window=self.max_window,
                q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. KDJ 过滤 —— 双重条件
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])

        # 最近 max_window 根 K 线的 J 分位
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        # —— 2.5 60日均线条件（使用通用函数）
        hist["MA60"] = hist["close"].rolling(window=60, min_periods=1).mean()

        # 当前必须在 MA60 上方（保持原条件）
        if hist["close"].iloc[-1] < hist["MA60"].iloc[-1]:
            return False

        # 寻找最近一次“有效上穿 MA60”的 T（使用 max_window 作为回看长度，避免过旧）
        t_pos = last_valid_ma_cross_up(hist["close"], hist["MA60"], lookback_n=self.max_window)
        if t_pos is None:
            return False

            # 3. MACD：DIF > 0
        hist["DIF"] = compute_dif(hist)
        if hist["DIF"].iloc[-1] <= 0:
            return False

        # 4. 当日：收盘>长期线 且 短期线>长期线
        if not zx_condition_at_positions(hist, require_close_gt_long=True, require_short_gt_long=True, pos=None):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
            self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 额外预留 20 根 K 线缓冲
            hist = hist.tail(self.max_window + 20)
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class SuperB1Selector:
    """SuperB1 选股器

    过滤逻辑概览
    ----------------
    1. **历史匹配 (t_m)** — 在 *lookback_n* 个交易日窗口内，至少存在一日
       满足 :class:`BBIKDJSelector`。

    2. **盘整区间** — 区间 ``[t_m, date-1]`` 收盘价波动率不超过 ``close_vol_pct``。

    3. **当日下跌** — ``(close_{date-1} - close_date) / close_{date-1}``
       ≥ ``price_drop_pct``。

    4. **J 值极低** — ``J < j_threshold`` *或* 位于历史 ``j_q_threshold`` 分位。
    """

    # ---------------------------------------------------------------------
    # 构造函数
    # ---------------------------------------------------------------------
    def __init__(
            self,
            *,
            lookback_n: int = 60,
            close_vol_pct: float = 0.05,
            price_drop_pct: float = 0.03,
            j_threshold: float = -5,
            j_q_threshold: float = 0.10,
            # ↓↓↓ 新增：嵌套 BBIKDJSelector 配置
            B1_params: Optional[Dict[str, Any]] = None
    ) -> None:
        # ---------- 参数合法性检查 ----------
        if lookback_n < 2:
            raise ValueError("lookback_n 应 ≥ 2")
        if not (0 < close_vol_pct < 1):
            raise ValueError("close_vol_pct 应位于 (0, 1) 区间")
        if not (0 < price_drop_pct < 1):
            raise ValueError("price_drop_pct 应位于 (0, 1) 区间")
        if not (0 <= j_q_threshold <= 1):
            raise ValueError("j_q_threshold 应位于 [0, 1] 区间")
        if B1_params is None:
            raise ValueError("bbi_params没有给出")

        # ---------- 基本参数 ----------
        self.lookback_n = lookback_n
        self.close_vol_pct = close_vol_pct
        self.price_drop_pct = price_drop_pct
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold

        # ---------- 内部 BBIKDJSelector ----------
        self.bbi_selector = BBIKDJSelector(**(B1_params or {}))

        # 为保证给 BBIKDJSelector 提供足够历史，预留额外缓冲
        self._extra_for_bbi = self.bbi_selector.max_window + 20

    # 单支股票过滤核心
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if len(hist) < 2:
            return False

        # —— 新增：所有战法统一当日过滤
        if not passes_day_constraints_today(hist):
            return False

        # ---------- Step-0: 数据量判断 ----------
        if len(hist) < self.lookback_n + self._extra_for_bbi:
            return False

        # ---------- Step-1: 搜索满足 BBIKDJ 的 t_m ----------
        lb_hist = hist.tail(self.lookback_n + 1)  # +1 以排除自身
        tm_idx: int or None = None
        for idx in lb_hist.index[:-1]:
            if self.bbi_selector._passes_filters(hist.loc[:idx]):
                tm_idx = idx
                stable_seg = hist.loc[tm_idx: hist.index[-2], "close"]
                if len(stable_seg) < 3:
                    tm_idx = None
                    break
                high, low = stable_seg.max(), stable_seg.min()
                if low <= 0 or (high / low - 1) > self.close_vol_pct:
                    tm_idx = None
                    continue
                else:
                    break
        if tm_idx is None:
            return False

        # —— 新增：在 t_m 当日检查【收盘>长期线 且 短期线>长期线】
        tm_pos = hist.index.get_loc(tm_idx)
        if not zx_condition_at_positions(hist, require_close_gt_long=True, require_short_gt_long=True, pos=tm_pos):
            return False

        # ---------- Step-3: 当日相对前一日跌幅 ----------
        close_today, close_prev = hist["close"].iloc[-1], hist["close"].iloc[-2]
        if close_prev <= 0 or (close_prev - close_today) / close_prev < self.price_drop_pct:
            return False

        # ---------- Step-4: J 值极低 ----------
        kdj = compute_kdj(hist)
        j_today = float(kdj["J"].iloc[-1])
        j_window = kdj["J"].iloc[-self.lookback_n:].dropna()
        j_q_val = float(j_window.quantile(self.j_q_threshold)) if not j_window.empty else np.nan
        if not (j_today < self.j_threshold or j_today <= j_q_val):
            return False

        # —— 当日仅要求【短期线>长期线】
        if not zx_condition_at_positions(hist, require_close_gt_long=False, require_short_gt_long=True, pos=None):
            return False

        return True

    # 批量选股接口
    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        min_len = self.lookback_n + self._extra_for_bbi

        for code, df in data.items():
            hist = df[df["date"] <= date].tail(min_len)
            if len(hist) < min_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)

        return picks


class PeakKDJSelector:
    """
    Peaks + KDJ 选股器    
    """

    def __init__(
            self,
            j_threshold: float = -5,
            max_window: int = 90,
            fluc_threshold: float = 0.03,
            gap_threshold: float = 0.02,
            j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.max_window = max_window
        self.fluc_threshold = fluc_threshold  # 当日↔peak_(t-n) 波动率上限
        self.gap_threshold = gap_threshold  # oc_prev 必须高于区间最低收盘价的比例
        self.j_q_threshold = j_q_threshold

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False

        if not passes_day_constraints_today(hist):
            return False

        hist = hist.copy().sort_values("date")
        hist["oc_max"] = hist[["open", "close"]].max(axis=1)

        # 1. 提取 peaks
        peaks_df = _find_peaks(
            hist,
            column="oc_max",
            distance=6,
            prominence=0.5,
        )

        # 至少两个峰      
        date_today = hist.iloc[-1]["date"]
        peaks_df = peaks_df[peaks_df["date"] < date_today]
        if len(peaks_df) < 2:
            return False

        peak_t = peaks_df.iloc[-1]  # 最新一个峰
        peaks_list = peaks_df.reset_index(drop=True)
        oc_t = peak_t.oc_max
        total_peaks = len(peaks_list)

        # 2. 回溯寻找 peak_(t-n)
        target_peak = None
        for idx in range(total_peaks - 2, -1, -1):
            peak_prev = peaks_list.loc[idx]
            oc_prev = peak_prev.oc_max
            if oc_t <= oc_prev:  # 要求 peak_t > peak_(t-n)
                continue

            # 只有当“总峰数 ≥ 3”时才检查区间内其他峰 oc_max
            if total_peaks >= 3 and idx < total_peaks - 2:
                inter_oc = peaks_list.loc[idx + 1: total_peaks - 2, "oc_max"]
                if not (inter_oc < oc_prev).all():
                    continue

            # 新增： oc_prev 高于区间最低收盘价 gap_threshold
            date_prev = peak_prev.date
            mask = (hist["date"] > date_prev) & (hist["date"] < peak_t.date)
            min_close = hist.loc[mask, "close"].min()
            if pd.isna(min_close):
                continue  # 区间无数据
            if oc_prev <= min_close * (1 + self.gap_threshold):
                continue

            target_peak = peak_prev

            break

        if target_peak is None:
            return False

        # 3. 当日收盘价波动率
        close_today = hist.iloc[-1]["close"]
        fluc_pct = abs(close_today - target_peak.close) / target_peak.close
        if fluc_pct > self.fluc_threshold:
            return False

        # 4. KDJ 过滤
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        if not zx_condition_at_positions(hist, require_close_gt_long=True, require_short_gt_long=True, pos=None):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
            self,
            date: pd.Timestamp,
            data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            hist = hist.tail(self.max_window + 20)  # 额外缓冲
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BBIShortLongSelector:
    """
    BBI 上升 + 短/长期 RSV 条件 + DIF > 0 选股器
    """

    def __init__(
            self,
            n_short: int = 3,
            n_long: int = 21,
            m: int = 3,
            bbi_min_window: int = 90,
            max_window: int = 150,
            bbi_q_threshold: float = 0.05,
            upper_rsv_threshold: float = 75,
            lower_rsv_threshold: float = 25
    ) -> None:
        if m < 2:
            raise ValueError("m 必须 ≥ 2")
        self.n_short = n_short
        self.n_long = n_long
        self.m = m
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.bbi_q_threshold = bbi_q_threshold
        self.upper_rsv_threshold = upper_rsv_threshold
        self.lower_rsv_threshold = lower_rsv_threshold

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        if not passes_day_constraints_today(hist):
            return False

            # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
                hist["BBI"],
                min_window=self.bbi_min_window,
                max_window=self.max_window,
                q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. 计算短/长期 RSV -----------------
        hist["RSV_short"] = compute_rsv(hist, self.n_short)
        hist["RSV_long"] = compute_rsv(hist, self.n_long)

        if len(hist) < self.m:
            return False  # 数据不足

        win = hist.iloc[-self.m:]  # 最近 m 天
        long_ok = (win["RSV_long"] >= self.upper_rsv_threshold).all()  # 长期 RSV 全 ≥ upper_rsv_threshold

        short_series = win["RSV_short"]

        # 条件：从最近 m 天的第一天起，存在某天 i 满足 RSV_short[i] >= upper，
        # 且在该天之后（j > i）存在某天 j 满足 RSV_short[j] < lower
        mask_upper = short_series >= self.upper_rsv_threshold
        mask_lower = short_series < self.lower_rsv_threshold

        has_upper_then_lower = False
        if mask_upper.any():
            upper_indices = np.where(mask_upper.to_numpy())[0]
            for i in upper_indices:
                # 只检查 i 之后的日子
                if i + 1 < len(short_series) and mask_lower.iloc[i + 1:].any():
                    has_upper_then_lower = True
                    break

        end_ok = short_series.iloc[-1] >= self.upper_rsv_threshold

        if not (long_ok and has_upper_then_lower and end_ok):
            return False

        # 3. MACD：DIF > 0 -------------------
        hist["DIF"] = compute_dif(hist)
        if hist["DIF"].iloc[-1] <= 0:
            return False

        # 4. 新增：知行情形
        if not zx_condition_at_positions(hist, require_close_gt_long=True, require_short_gt_long=True, pos=None):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
            self,
            date: pd.Timestamp,
            data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 预留足够长度：RSV 计算窗口 + BBI 检测窗口 + m
            need_len = (
                    max(self.n_short, self.n_long)
                    + self.bbi_min_window
                    + self.m
            )
            hist = hist.tail(max(need_len, self.max_window))
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class MA60CrossVolumeWaveSelector:
    """
    条件：
    1) 当日 J 绝对低或相对低（J < j_threshold 或 J ≤ 近 max_window 根 J 的 j_q_threshold 分位）
    2) 最近 lookback_n 内，存在一次“有效上穿 MA60”（t-1 收盘 < MA60, t 收盘 ≥ MA60）；
       且从该上穿日 T 到今天的“上涨波段”日均成交量 ≥ 上穿前等长窗口的日均成交量 * vol_multiple
       —— 上涨波段定义为 [T, today] 间的所有交易日（不做趋势单调性强约束，稳健且可复现）
    3) 近 ma60_slope_days（默认 5）个交易日的 MA60 回归斜率 > 0
    """

    def __init__(
            self,
            *,
            lookback_n: int = 60,
            vol_multiple: float = 1.5,
            j_threshold: float = -5.0,
            j_q_threshold: float = 0.10,
            ma60_slope_days: int = 5,
            max_window: int = 120,  # 用于计算 J 分位
    ) -> None:
        if lookback_n < 2:
            raise ValueError("lookback_n 应 ≥ 2")
        if not (0.0 <= j_q_threshold <= 1.0):
            raise ValueError("j_q_threshold 应位于 [0,1]")
        if ma60_slope_days < 2:
            raise ValueError("ma60_slope_days 应 ≥ 2")
        self.lookback_n = lookback_n
        self.vol_multiple = vol_multiple
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold
        self.ma60_slope_days = ma60_slope_days
        self.max_window = max_window

    @staticmethod
    def _ma_slope_positive(series: pd.Series, days: int) -> bool:
        """对最近 days 个点做一阶线性回归，斜率 > 0 判为正"""
        seg = series.dropna().tail(days)
        if len(seg) < days:
            return False
        x = np.arange(len(seg), dtype=float)
        # 线性回归（最小二乘）：斜率 k
        k, _ = np.polyfit(x, seg.values.astype(float), 1)
        return bool(k > 0)

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        """
        hist：按日期升序，最后一行是目标交易日
        需包含列：date, open, high, low, close, volume
        """
        if hist.empty:
            return False

        hist = hist.copy().sort_values("date")
        # 至少要有 60 日用于 MA60，再加 lookback/slope 的缓冲
        min_len = max(60 + self.lookback_n + self.ma60_slope_days, self.max_window + 5)
        if len(hist) < min_len:
            return False

        if not passes_day_constraints_today(hist):
            return False

        # --- 计算指标 ---
        kdj = compute_kdj(hist)
        j_today = float(kdj["J"].iloc[-1])
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_q_val = float(j_window.quantile(self.j_q_threshold))

        # 1) 当日 J 绝对低或相对低
        if not (j_today < self.j_threshold or j_today <= j_q_val):
            return False

        # 2) MA60 及有效上穿（使用通用函数）
        hist["MA60"] = hist["close"].rolling(window=60, min_periods=1).mean()
        if hist["close"].iloc[-1] < hist["MA60"].iloc[-1]:
            return False

        t_pos = last_valid_ma_cross_up(hist["close"], hist["MA60"], lookback_n=self.lookback_n)
        if t_pos is None:
            return False

        # === [T, today] 内以 High 最大值的交易日为 Tmax ===
        seg_T_to_today = hist.iloc[t_pos:]
        if seg_T_to_today.empty:
            return False

        # 若并列最高，默认取“第一次”出现的那天；要“最后一次”可改见注释
        tmax_label = seg_T_to_today["high"].idxmax()
        int_pos_T = t_pos
        int_pos_Tmax = hist.index.get_loc(tmax_label)

        if int_pos_Tmax < int_pos_T:
            return False

        # 上涨波段 [T, Tmax]（含端点）
        wave = hist.iloc[int_pos_T: int_pos_Tmax + 1]
        wave_len = len(wave)
        if wave_len < 3:
            return False

        # 等长前置窗口 [T - wave_len, T-1]
        pre_start_pos = max(0, int_pos_T - min(wave_len, 10))
        pre = hist.iloc[pre_start_pos:int_pos_T]
        if len(pre) < max(5, min(10, wave_len)):
            return False

        # 成交量均值对比
        wave_avg_vol = float(wave["volume"].replace(0, np.nan).dropna().mean())
        pre_avg_vol = float(pre["volume"].replace(0, np.nan).dropna().mean())
        if not (np.isfinite(wave_avg_vol) and np.isfinite(pre_avg_vol) and pre_avg_vol > 0):
            return False

        if wave_avg_vol < self.vol_multiple * pre_avg_vol:
            return False

        # 3) MA60 斜率 > 0（保留原实现）
        if not self._ma_slope_positive(hist["MA60"], self.ma60_slope_days):
            return False

        if not zx_condition_at_positions(hist, require_close_gt_long=True, require_short_gt_long=True, pos=None):
            return False

        return True

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        # 给足 60 日均线与量能比较的历史长度
        need_len = max(60 + self.lookback_n + self.ma60_slope_days, self.max_window + 20)
        for code, df in data.items():
            hist = df[df["date"] <= date].tail(need_len)
            if len(hist) < need_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)
        return picks


class BigBullishVolumeSelector:

    def __init__(
            self,
            *,
            up_pct_threshold: float = 0.04,  # 长阳阈值：例如 0.04 表示涨幅>4%
            upper_wick_pct_max: float = 0.5,  # 上影线比例上限（口径由 wick_mode 决定）
            vol_lookback_n: int = 20,  # 放量比较的历史天数 n
            vol_multiple: float = 1.5,  # 放量倍数阈值
            min_history: int or None = None,  # 最少历史长度（默认自动 = vol_lookback_n + 2）
            require_bullish_close: bool = True,  # 可选：要求当日收阳（close >= open）
            ignore_zero_volume: bool = True,  # 计算均量时是否忽略 volume=0
            close_lt_zxdq_mult: float = 1.0  # 例如 1.0 表示 close < zxdq；1.02 表示 close < 1.02*zxdq
    ) -> None:
        if up_pct_threshold <= 0:
            raise ValueError("up_pct_threshold 应 > 0")
        if upper_wick_pct_max < 0:
            raise ValueError("upper_wick_pct_max 应 >= 0")
        if vol_lookback_n < 1:
            raise ValueError("vol_lookback_n 应 >= 1")
        if vol_multiple <= 0:
            raise ValueError("vol_multiple 应 > 0")
        if close_lt_zxdq_mult <= 0:
            raise ValueError("close_lt_zxdq_mult 应 > 0")

        self.up_pct_threshold = float(up_pct_threshold)
        self.upper_wick_pct_max = float(upper_wick_pct_max)
        self.vol_lookback_n = int(vol_lookback_n)
        self.vol_multiple = float(vol_multiple)
        self.require_bullish_close = bool(require_bullish_close)
        self.ignore_zero_volume = bool(ignore_zero_volume)
        self.close_lt_zxdq_mult = float(close_lt_zxdq_mult)
        self.eps = float(1e-12)
        self.min_history = int(min_history) if min_history is not None else (self.vol_lookback_n + 2)

    @staticmethod
    def _to_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _upper_wick_pct(self, o: float, h: float, c: float) -> float:
        return (h - max(o, c)) / max(o, c)

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if hist is None or hist.empty:
            return False

        hist = hist.sort_values("date").copy()

        if len(hist) < self.min_history:
            return False
        if len(hist) < (self.vol_lookback_n + 2):
            return False  # 至少需要：T、T-1、以及 T-1 往前 n 天

        today = hist.iloc[-1]
        prev = hist.iloc[-2]

        oT = self._to_float(today.get("open"))
        hT = self._to_float(today.get("high"))
        lT = self._to_float(today.get("low"))
        cT = self._to_float(today.get("close"))
        vT = self._to_float(today.get("volume"))

        cP = self._to_float(prev.get("close"))

        # 基础合法性
        if not (np.isfinite(oT) and np.isfinite(hT) and np.isfinite(lT) and np.isfinite(cT) and np.isfinite(
                vT) and np.isfinite(cP)):
            return False
        if cP <= 0 or cT <= 0:
            return False
        if hT < max(oT, cT) or lT > min(oT, cT):
            # K线数据异常（不一定必需，但建议保持严谨）
            return False

        # (可选) 要求当日收阳
        if self.require_bullish_close and not (cT >= oT):
            return False

        # 1) 长阳：涨幅 > 阈值
        pct_chg = cT / cP - 1.0
        if pct_chg <= self.up_pct_threshold:
            return False

        # 2) 上影线百分比 < 阈值
        wick_pct = self._upper_wick_pct(oT, hT, cT)
        if not np.isfinite(wick_pct):
            return False
        if wick_pct >= self.upper_wick_pct_max:
            return False

        # 3) 放量：当日成交量 > 前 n 日均量 * 倍数
        vol_hist = hist["volume"].iloc[-(self.vol_lookback_n + 1):-1].astype(float)  # T-n ... T-1
        if self.ignore_zero_volume:
            vol_hist = vol_hist.replace(0, np.nan).dropna()

        if len(vol_hist) < max(3, int(self.vol_lookback_n * 0.6)):
            # 有效样本过少就不做判断（你也可以改成直接 False 或严格要求=vol_lookback_n）
            return False

        avg_vol = float(vol_hist.mean())
        if not (np.isfinite(avg_vol) and avg_vol > 0):
            return False

        if vT < self.vol_multiple * avg_vol:
            return False

        # 4) 偏离短线小于阈值
        try:
            zxdq, _ = compute_zx_lines(hist)
            zxdq_T = float(zxdq.iloc[-1])
        except Exception:
            zxdq_T = float("nan")

        if not np.isfinite(zxdq_T):
            return False
        else:
            if not (cT < zxdq_T * self.close_lt_zxdq_mult):
                return False

        return True

    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks: List[str] = []
        need_len = max(self.min_history, self.vol_lookback_n + 2)

        for code, df in data.items():
            if df is None or df.empty:
                continue
            hist = df[df["date"] <= date].tail(need_len)
            if len(hist) < need_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)

        return picks


# ========== 日志配置（控制台+文件） ==========
log_dir = "选股日志"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"红梅B1战法选股日志_{datetime.now().strftime('%Y%m%d')}.log")

logger = logging.getLogger("HongMeiB1Selector")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# 控制台输出
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)

# 文件输出（UTF-8防乱码）
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class HongMeiB1Selector:
    """
    红梅师兄 B1 战法 —— 通达信 100% 等效复刻版（带详细日志补全版）
    """

    def __init__(self,
                 j_threshold: float = 13.0,
                 yangyin_mult: float = 1.5,
                 a28_threshold: float = 0.005,
                 mv_threshold: float = 50.0,
                 avg40_vol_mult: float = 1.8,
                 plry_cnt_min: int = 3,
                 bigv_mult: float = 1.75,
                 m1: int = 14,
                 m2: int = 28,
                 m3: int = 57,
                 m4: int = 114):

        self.j_threshold = j_threshold
        self.yangyin_mult = yangyin_mult
        self.a28_threshold = a28_threshold
        self.mv_threshold = mv_threshold
        self.avg40_vol_mult = avg40_vol_mult
        self.plry_cnt_min = plry_cnt_min
        self.bigv_mult = bigv_mult
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4

    # ===== 通达信 SMA 算法：递归对齐 =====
    def _sma(self, s: pd.Series, n: int, m: int = 1):
        s = s.fillna(0)
        res = np.zeros_like(s)
        # 寻找第一个有效值起始点
        start_idx = (s != 0).argmax() if (s != 0).any() else 0
        res[start_idx] = s.iloc[start_idx]
        for i in range(start_idx + 1, len(s)):
            res[i] = (m * s.iloc[i] + (n - m) * res[i - 1]) / n
        return pd.Series(res, index=s.index)

    # ===== 通达信 EMA 算法：用于知行趋势线 =====
    def _ema(self, s: pd.Series, n: int):
        return s.ewm(span=n, adjust=False).mean()

    def _passes_filters(self, df, code):
        if df is None or len(df) < 120:
            logger.info(f"[{code}] 跳过：历史数据不足 {len(df)}")
            return False

        df = df.copy().sort_values('date').reset_index(drop=True)
        C, O, H, L = df['close'], df['open'], df['high'], df['low']
        V, AMT, CAP = df['volume'], df['amount'], df['capital']

        # ===== 1. KDJ 计算 =====
        low_9 = L.rolling(9, min_periods=1).min()
        high_9 = H.rolling(9, min_periods=1).max()
        den = high_9 - low_9
        rsv = pd.Series(np.where(den == 0, 50, (C - low_9) / den * 100), index=df.index)

        K = self._sma(rsv, 3, 1)
        D = self._sma(K, 3, 1)
        J = 3 * K - 2 * D
        j_val = J.iloc[-1]

        if j_val > self.j_threshold:
            logger.info(f"[{code}] 跳过：J值不达标 {j_val:.2f}")
            return False
        logger.info(f"[{code}] J值通过 {j_val:.2f}")

        # ===== 2. 阳阴量比 (REAL_YANG 精确对齐) =====
        # 通达信: REAL_YANG:=C>O AND NOT(C<REF(C,1)); 即 C>O 且 C>=REF(C,1)
        df['REAL_YANG'] = (C > O) & (C >= C.shift(1))
        df['REAL_YIN'] = (C < O) & (C <= C.shift(1))

        vy21 = (V * df['REAL_YANG']).rolling(21, min_periods=1).sum()
        vi21 = (V * df['REAL_YIN']).rolling(21, min_periods=1).sum()
        vy14 = (V * df['REAL_YANG']).rolling(14, min_periods=1).sum()
        vi14 = (V * df['REAL_YIN']).rolling(14, min_periods=1).sum()

        y21_ratio = vy21.iloc[-1] / max(vi21.iloc[-1], 1e-6)
        y14_ratio = vy14.iloc[-1] / max(vi14.iloc[-1], 1e-6)

        yangyin_ok = (vy21.iloc[-1] > self.yangyin_mult * vi21.iloc[-1]) or \
                     (vy14.iloc[-1] > self.yangyin_mult * vi14.iloc[-1])

        if not yangyin_ok:
            logger.info(f"[{code}] 跳过：阳阴量比不达标 21:{y21_ratio:.2f} 14:{y14_ratio:.2f}")
            return False
        logger.info(f"[{code}] 阳阴量比通过 21:{y21_ratio:.2f} 14:{y14_ratio:.2f}")

        # ===== 3. A28 (成交额过滤) 修正版 =====
        # Tushare amount 单位是千元，通达信 AMOUNT 是元
        # 换算成亿元：(千元 * 1000) / 1e8 = 千元 / 1e5
        a28_val = (AMT.rolling(28, min_periods=1).mean() / 1e5).iloc[-1]

        if a28_val < self.a28_threshold:
            logger.info(f"[{code}] 跳过：A28不达标 {a28_val:.6f}")
            return False
        logger.info(f"[{code}] A28通过 {a28_val:.6f}")

        # ===== 4. 市值 (MV) 修正版 =====
        # Tushare capital 单位是万股，C 是元
        # 换算成亿元：(元 * 万股) / 10000
        mv_val = (C.iloc[-1] * CAP.iloc[-1]) / 10000

        if mv_val < self.mv_threshold:
            logger.info(f"[{code}] 跳过：市值不达标 {mv_val:.2f}亿")
            return False
        logger.info(f"[{code}] 市值通过 {mv_val:.2f}亿")

        # ===== 5. GOOD28 (高位放量阴线过滤) =====
        o_llv28 = O.rolling(28, min_periods=1).min()
        o_hhv28 = O.rolling(28, min_periods=1).max()
        o85 = o_llv28 + 0.925 * (o_hhv28 - o_llv28)

        top15o = O >= o85
        fd15 = (C < C.shift(1)) & (C <= O) & (V >= 1.15 * V.shift(1))
        bad_cnt = (top15o & fd15).rolling(28, min_periods=1).sum().iloc[-1]

        if bad_cnt != 0:
            logger.info(f"[{code}] 跳过：GOOD28不达标 (放量阴线次数={int(bad_cnt)})")
            return False
        logger.info(f"[{code}] GOOD28通过")

        # ===== 6. MAX28 (最高成交量不应是阴线) =====
        maxvol28 = V.rolling(28, min_periods=1).max()
        max_yin_cnt = ((V == maxvol28) & df['REAL_YIN']).rolling(28, min_periods=1).sum().iloc[-1]

        if max_yin_cnt != 0:
            logger.info(f"[{code}] 跳过：MAX28不达标 (最大量日为 REAL_YIN)")
            return False
        logger.info(f"[{code}] MAX28通过")

        # ===== 7. PLRY (倍量柱统计) =====
        avg40_v = V.rolling(40, min_periods=1).mean()
        plry = (V > self.avg40_vol_mult * V.shift(1)) & (C > O) & (V > avg40_v)
        plry_cnt = plry.rolling(28, min_periods=1).sum().iloc[-1]
        plry_ok = plry_cnt >= self.plry_cnt_min

        # ===== 8. TRIGGER (关键K形态) =====
        v40p = V.shift(1).rolling(40, min_periods=1).mean()
        bd = (C > C.shift(1)) & (C >= O)
        bigv = V > self.bigv_mult * v40p

        c_llv40 = C.rolling(40, min_periods=1).min()
        c_hhv40 = C.rolling(40, min_periods=1).max()
        r55 = c_llv40 + 0.55 * (c_hhv40 - c_llv40)
        posok = C > r55

        trigger = plry_ok or (bd.iloc[-1] and bigv.iloc[-1] and posok.iloc[-1])

        logger.info(
            f"[{code}] TRIGGER分析: PLRY_OK={plry_ok}({int(plry_cnt)}次), BD={bd.iloc[-1]}, BIGV={bigv.iloc[-1]}, POSOK={posok.iloc[-1]}")

        if not trigger:
            logger.info(f"[{code}] 跳过：TRIGGER未激活")
            return False

        # ===== 9. 知行趋势线 (WL > YL 且 C > YL) =====
        wl = self._ema(self._ema(C, 10), 10)
        yl = (C.rolling(self.m1, min_periods=1).mean() +
              C.rolling(self.m2, min_periods=1).mean() +
              C.rolling(self.m3, min_periods=1).mean() +
              C.rolling(self.m4, min_periods=1).mean()) / 4

        cond_wl = wl.iloc[-1] > yl.iloc[-1]
        cond_yl = C.iloc[-1] > yl.iloc[-1]

        if not (cond_wl and cond_yl):
            logger.info(f"[{code}] 跳过：趋势线不达标 WL>YL:{cond_wl}, C>YL:{cond_yl}")
            return False

        logger.info(f"[{code}] ✅ 完全满足通达信B1条件")
        return True

    def select(self, date: pd.Timestamp, data: dict) -> list:
        picks = []
        logger.info(f"\n========== 红梅B1 启动筛选 [{date.strftime('%Y-%m-%d')}] ==========")
        for code, df in data.items():
            try:
                sub = df[df['date'] <= date].copy()
                if self._passes_filters(sub, code):
                    picks.append(code)
            except Exception as e:
                logger.error(f"[{code}] 筛选异常: {e}")
        return picks


class BrickMomentumSelector:
    """
    砖头动能增强策略 —— 基于知行趋势线 + 砖型图力度过滤
    """

    def __init__(self,
                 m1=14, m2=28, m3=57, m4=114,
                 brick_threshold=4,
                 strength_ratio=0.666):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.brick_threshold = brick_threshold
        self.strength_ratio = strength_ratio

    def _sma(self, s: pd.Series, n: int, m: int = 1):
        """通达信专用递归SMA算法"""
        s = s.fillna(0)
        res = np.zeros_like(s)
        start_idx = (s != 0).argmax() if (s != 0).any() else 0
        res[start_idx] = s.iloc[start_idx]
        for i in range(start_idx + 1, len(s)):
            res[i] = (m * s.iloc[i] + (n - m) * res[i - 1]) / n
        return pd.Series(res, index=s.index)

    def _ema(self, s: pd.Series, n: int):
        """通达信专用EMA算法"""
        return s.ewm(span=n, adjust=False).mean()

    def _passes_filters(self, df, code):
        """
        2.0倍以上红砖动力 + [0, 3%]涨幅控制 + 趋势共振
        """
        # 1. 基础数据校验：确保数据量足以计算最长均线 M4(114)
        if df is None or len(df) < 120:
            return False

        # 确保按日期升序排列
        df = df.copy().sort_values('date').reset_index(drop=True)
        C, H, L = df['close'], df['high'], df['low']

        # ===== 2. 趋势线判定 (白线要在黄线上方) =====
        wl = self._ema(self._ema(C, 10), 10)
        # 计算黄线时，若数据不足 m1/m2/m3/m4，rolling 会返回 NaN，逻辑会自动过滤掉
        yl = (C.rolling(self.m1).mean() +
              C.rolling(self.m2).mean() +
              C.rolling(self.m3).mean() +
              C.rolling(self.m4).mean()) / 4

        if wl.iloc[-1] <= yl.iloc[-1]:
            return False

        # ===== 3. 砖型图核心算法 =====
        hhv4 = H.rolling(4).max()
        llv4 = L.rolling(4).min()
        diff4 = hhv4 - llv4

        var1a = (hhv4 - C) / np.where(diff4 == 0, 1e-6, diff4) * 100 - 90
        var2a = self._sma(var1a, 4, 1) + 100

        var3a = (C - llv4) / np.where(diff4 == 0, 1e-6, diff4) * 100
        var4a = self._sma(var3a, 6, 1)
        var5a = self._sma(var4a, 6, 1) + 100

        var6a = var5a - var2a
        brick = np.where(var6a > 4, var6a - 4, 0)
        brick_series = pd.Series(brick, index=df.index)
        brick_diff = brick_series.diff()

        # ===== 4. 拐点与 2.0倍 力度判定 =====
        today_diff = brick_diff.iloc[-1]  # 今日红砖增量 (正数)
        yesterday_diff = brick_diff.iloc[-2]  # 昨日绿砖减量 (负数)

        is_reversal = (today_diff > 0) and (yesterday_diff < 0)
        if not is_reversal:
            return False

        strength_ok = today_diff >= (abs(yesterday_diff) * 2.0)
        if not strength_ok:
            return False

        # ===== 5. 涨幅控制 (寻找低位起步点) =====
        last_close = C.iloc[-1]
        prev_close = C.iloc[-2]
        change_pct = (last_close - prev_close) / prev_close * 100

        if not (0 < change_pct <= 3.0):
            return False

        # ===== 6. 最终记录与通过 =====
        logger.info(
            f"[{code}] 🚀 捕捉到2倍动力反转！力度:{today_diff / abs(yesterday_diff):.2f}x | 涨幅:{change_pct:.2f}%")

        return True

    def select(self, date: pd.Timestamp, data: dict) -> list:
        picks = []
        # 统一日期格式为字符串，修复 '<=' not supported between 'str' and 'Timestamp' 报错
        target_date_str = date.strftime('%Y-%m-%d')
        logger.info(f"--- 砖头动能策略开始筛选 [{target_date_str}] ---")

        for code, df in data.items():
            try:
                # 确保 df 内部日期也是字符串格式
                df['date'] = df['date'].astype(str)
                sub = df[df['date'] <= target_date_str].copy()
                if self._passes_filters(sub, code):
                    picks.append(code)
            except Exception as e:
                logger.error(f"[{code}] 策略运行异常: {e}")
        return picks


class HongMeiB1BrickCombinedSelector:
    """
    红梅B1 + 砖头爆发实战放宽版
    """

    def __init__(self,
                 m1=14, m2=28, m3=57, m4=114,
                 j_threshold=13.0,
                 strength_ratio=0.666):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.j_threshold = j_threshold
        self.strength_ratio = strength_ratio

    def _sma(self, s: pd.Series, n: int, m: int = 1):
        """通达信专用递归SMA算法"""
        s = s.fillna(0)
        res = np.zeros_like(s)
        start_idx = (s != 0).argmax() if (s != 0).any() else 0
        res[start_idx] = s.iloc[start_idx]
        for i in range(start_idx + 1, len(s)):
            res[i] = (m * s.iloc[i] + (n - m) * res[i - 1]) / n
        return pd.Series(res, index=s.index)

    def _ema(self, s: pd.Series, n: int):
        """通达信专用EMA算法"""
        return s.ewm(span=n, adjust=False).mean()

    def _passes_filters(self, df, code):
        if df is None or len(df) < 120:
            return False

        df = df.copy().sort_values('date').reset_index(drop=True)

        C = df['close']
        H = df['high']
        L = df['low']
        O = df['open']
        V = df['volume'] if 'volume' in df.columns else df['vol']
        AMT = df['amount']

        # 1.1 J值 (取最后一位)
        hhv9 = H.rolling(9).max()
        llv9 = L.rolling(9).min()
        den = hhv9 - llv9
        rsv = np.where(den == 0, 50, (C - llv9) / den * 100)
        k = self._sma(pd.Series(rsv), 3, 1)
        d = self._sma(k, 3, 1)
        j_val = (3 * k - 2 * d).iloc[-1]
        j_ok = j_val <= self.j_threshold

        # 1.2 阳阴量能 (取最后一位)
        real_yang = (C > O) & (C >= C.shift(1))
        real_yin = (C < O) & (C <= C.shift(1))
        vol_yang21 = (V * real_yang).rolling(21).sum().iloc[-1]
        vol_yin21 = (V * real_yin).rolling(21).sum().iloc[-1]
        vol_yang14 = (V * real_yang).rolling(14).sum().iloc[-1]
        vol_yin14 = (V * real_yin).rolling(14).sum().iloc[-1]
        yangyin_ok = (vol_yang21 > 1.5 * vol_yin21) or (vol_yang14 > 1.5 * vol_yin14)

        # 1.4 风控过滤 (取最后一位)
        o_llv28 = O.rolling(28).min()
        o_hhv28 = O.rolling(28).max()
        o85 = o_llv28 + 0.925 * (o_hhv28 - o_llv28)
        top15o = O >= o85
        fd15 = (C < C.shift(1)) & (C <= O) & (V >= 1.15 * V.shift(1))
        good28 = (top15o & fd15).rolling(28).sum().iloc[-1] == 0

        max_vol28 = V.rolling(28).max()
        max28_ok = ((V == max_vol28) & real_yin).rolling(28).sum().iloc[-1] == 0

        # 1.5 触发逻辑
        avg40 = V.rolling(40).mean()
        plry = (V > 1.8 * V.shift(1)) & (C > O) & (V > avg40)
        plry_cnt = plry.rolling(28).sum().iloc[-1] >= 3
        v40p = V.shift(1).rolling(40).mean()
        bigv_bool = ((C > C.shift(1)) & (C >= O) & (V > 1.75 * v40p)).iloc[-1]
        r55 = C.rolling(40).min() + 0.55 * (C.rolling(40).max() - C.rolling(40).min())
        pos_ok = C.iloc[-1] > r55.iloc[-1]

        b1_trigger = plry_cnt or (bigv_bool and pos_ok)
        b1_base = b1_trigger and j_ok and (AMT.iloc[-1] > 0) and good28 and max28_ok and yangyin_ok

        # 2. 趋势 (取最后一位)
        wl = self._ema(self._ema(C, 10), 10).iloc[-1]
        yl = ((C.rolling(self.m1).mean() + C.rolling(self.m2).mean() +
               C.rolling(self.m3).mean() + C.rolling(self.m4).mean()) / 4).iloc[-1]
        trend_ok = (wl > yl) and (C.iloc[-1] > yl)

        # 3. 砖头 (取最后两位计算增量)
        hhv4 = H.rolling(4).max()
        llv4 = L.rolling(4).min()
        diff4 = hhv4 - llv4
        var1a = (hhv4 - C) / np.where(diff4 == 0, 1e-6, diff4) * 100 - 90
        var2a = self._sma(var1a, 4, 1) + 100
        var3a = (C - llv4) / np.where(diff4 == 0, 1e-6, diff4) * 100
        var4a = self._sma(var3a, 6, 1)
        var5a = self._sma(var4a, 6, 1) + 100
        var6a = var5a - var2a
        brick = np.where(var6a > 4, var6a - 4, 0)
        brick_diff = pd.Series(brick).diff()

        t_diff = brick_diff.iloc[-1]
        y_diff = brick_diff.iloc[-2]
        brick_ok = (t_diff > 0) and (y_diff < 0) and (t_diff >= abs(y_diff) * self.strength_ratio)

        return bool(b1_base and trend_ok and brick_ok)

    def select(self, date: pd.Timestamp, data: dict) -> list:
        picks = []
        # 将输入日期转换为字符串，方便与 DataFrame 比较
        target_date_str = date.strftime('%Y-%m-%d')
        logger.info(f"--- 红梅B1+砖头策略筛选中 [{target_date_str}] ---")

        for code, df in data.items():
            try:
                df['date'] = df['date'].astype(str)
                sub = df[df['date'] <= target_date_str].copy()
                if self._passes_filters(sub, code):
                    picks.append(code)
            except Exception as e:
                logger.error(f"[{code}] 策略运行异常: {e}")
        return picks


class SingleNeedleUnder20Selector:
    """
    单针下20 - 通达信对标版
    N1:=3;
    短期:=100*(C-LLV(L,N1))/(HHV(C,N1)-LLV(L,N1));
    选股: REF(短期,1)>=80 AND 短期<=20;
    """

    def __init__(self, n1=3):
        self.n1 = n1

    def _passes_filters(self, df, code):
        # 确保数据按日期从旧到新排列
        df = df.copy().sort_values('date').reset_index(drop=True)

        # 基础数据量检查：N1周期 + 1位REF，至少需要 N1+1 行有效数据
        if len(df) < self.n1 + 1:
            return False

        C = df['close']
        L = df['low']

        # --- 还原通达信指标计算 ---
        # 1. LLV(L, N1): N1周期内最低价的最低值
        llv_l = L.rolling(window=self.n1).min()

        # 2. HHV(C, N1): N1周期内收盘价的最高值 (注意公式里是C)
        hhv_c = C.rolling(window=self.n1).max()

        # 3. 计算“短期”
        diff = hhv_c - llv_l
        # 模拟通达信在分母为0时的处理（通常返回0或保持不变，这里用极小值规避溢出）
        short_term = 100 * (C - llv_l) / np.where(diff == 0, 1e-6, diff)

        # --- 执行选股条件 ---
        # 今天的指标: 短期
        current_val = short_term.iloc[-1]
        # 昨天的指标: REF(短期, 1)
        yesterday_val = short_term.iloc[-2]

        # 选股逻辑: REF(短期,1)>=80 AND 短期<=20
        is_hit = (yesterday_val >= 80) and (current_val <= 20)

        return bool(is_hit)

    def select(self, date: pd.Timestamp, data: dict) -> list:
        picks = []
        target_date_str = date.strftime('%Y-%m-%d')

        for code, df in data.items():
            try:
                # 过滤出目标日期及之前的所有历史数据
                df['date'] = df['date'].astype(str)
                sub = df[df['date'] <= target_date_str].copy()

                if self._passes_filters(sub, code):
                    picks.append(code)
            except Exception as e:
                logger.error(f"[{code}] 计算异常: {e}")
        return picks
