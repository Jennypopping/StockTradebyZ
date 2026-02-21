from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import requests
import tushare as ts
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 新增：股票名称映射工具函数 ----------
def load_stock_name_map(map_path: Path = Path("./stock_name_map.csv")) -> Dict[str, str]:
    """
    加载股票代码-中文名映射表（增加容错）
    :param map_path: 本地映射表CSV路径
    :return: {代码: 名称}的字典
    """
    name_map = {}
    # 方式1：加载本地CSV（增加容错）
    if map_path.exists():
        # 检查文件大小是否为0
        if map_path.stat().st_size == 0:
            logger.error(f"股票名称映射表 {map_path} 为空文件")
        else:
            try:
                # 兼容UTF-8/GBK编码，指定列头
                df = pd.read_csv(
                    map_path,
                    encoding="utf-8",  # 若文件是GBK，改为encoding="gbk"
                    header=0,  # 强制第一行为列头
                    on_bad_lines="skip"  # 跳过错误行
                )
                # 校验必要列
                if "code" not in df.columns or "name" not in df.columns:
                    logger.error(f"股票名称映射表缺少 'code' 或 'name' 列")
                else:
                    # 确保code列是字符串，避免格式问题
                    df['code'] = df['code'].astype(str).str.strip()
                    df['name'] = df['name'].astype(str).str.strip()
                    # 过滤空值
                    df = df.dropna(subset=["code", "name"])
                    name_map = dict(zip(df['code'], df['name']))
                    logger.info(f"加载了 {len(name_map)} 只股票的名称映射")
            except Exception as e:
                logger.error(f"读取股票名称映射表失败：{e}")

    # 方式2：备用方案 - 从tushare获取（需配置token）
    if not name_map:
        logger.warning("本地映射表加载失败，尝试从tushare获取...")
        try:
            ts_token = "26d6a5877ad3da85312145f9975b98873a5291b5a50e3af33d6b77671b70"
            pro = ts.pro_api(ts_token)
            pro._DataApi__token = ts_token  # 保证有这个代码，不然不可以获取
            pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
            # 数据清洗
            df['ts_code'] = df['ts_code'].astype(str).str.strip()
            df['name'] = df['name'].astype(str).str.strip()
            name_map = dict(zip(df['ts_code'], df['name']))
            # 保存到本地（确保格式正确）
            df.rename(columns={"ts_code": "code"}).to_csv(
                map_path, index=False, encoding="utf-8"
            )
            logger.info(f"从tushare获取并保存了 {len(name_map)} 只股票的名称映射")
        except Exception as e:
            logger.error(f"从tushare获取名称映射失败：{e}")
            return {}

    return name_map


def get_stock_name(code: str, name_map: Dict[str, str]) -> str:
    """
    根据代码获取中文名，自动补全后缀，无匹配时返回代码本身
    :param code: 股票代码（如301052 或 301052.SZ）
    :param name_map: 名称映射字典
    :return: 中文名或代码
    """
    # 步骤1：如果代码无后缀，自动补全.SZ/.SH
    if "." not in code:
        if code.startswith(("6", "9")):  # 沪市代码开头
            code = f"{code}.SH"
        elif code.startswith(("0", "3")):  # 深市代码开头
            code = f"{code}.SZ"

    # 步骤2：匹配名称
    return name_map.get(code, code)


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        frames[code] = df
    return frames


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    args = p.parse_args()

    # --- 新增：加载股票名称映射 ---
    name_map = load_stock_name_map(Path("./stock_name_map.csv"))

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all"
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    data = load_data(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    if args.date:
        trade_date = pd.to_datetime(args.date)
    else:
        valid_dates = []
        for df in data.values():
            if df.empty:
                continue
            d = df["date"].max()
            if isinstance(d, pd.Timestamp):
                valid_dates.append(d)
        trade_date = max(valid_dates)


    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # ========== 新增：初始化汇总数据 ==========
    summary_data = {
        "trade_date": trade_date.date(),
        "total_strategies": 0,
        "active_strategies": 0,
        "failed_strategies": [],
        "selector_results": []
    }

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        summary_data["total_strategies"] += 1  # 累计总策略数
        if cfg.get("activate", True) is False:
            continue

        summary_data["active_strategies"] += 1  # 累计激活策略数
        alias = cfg.get("alias", "未知策略")
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            error_info = f"{alias}：{str(e)}"
            logger.error("跳过配置 %s：%s", cfg, e)
            summary_data["failed_strategies"].append(error_info)
            continue

        picks = selector.select(trade_date, data)

        # ========== 新增：存储当前策略结果 ==========
        strategy_result = {
            "alias": alias,
            "stock_count": len(picks),
            "stocks": picks if picks else ["无符合条件股票"]
        }
        summary_data["selector_results"].append(strategy_result)

        # 原有日志输出逻辑（保留）
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        logger.info("%s", ", ".join(picks) if picks else "无符合条件股票")

    # ========== 新增：汇总并推送飞书 ==========
    # 1. 替换为你的飞书机器人Webhook地址
    FEISHU_WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/66fc005a-3dc9-4479-a6e2-6150d5860205"

    # 2. 格式化汇总内容
    push_content = format_summary(summary_data, name_map)

    # 3. 推送飞书
    send_feishu_message(FEISHU_WEBHOOK, push_content, msg_type="text")


def send_feishu_message(webhook_url, content, msg_type="text"):
    """
    飞书机器人推送消息
    :param webhook_url: 飞书机器人的Webhook地址
    :param content: 推送内容（text格式为字符串，card格式为字典）
    :param msg_type: 消息类型，可选"text"（文本）、"interactive"（卡片）
    :return: 是否推送成功
    """
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {
        "msg_type": msg_type,
        "content": {}
    }

    if msg_type == "text":
        payload["content"]["text"] = content
    elif msg_type == "interactive":
        payload["content"] = content  # 卡片格式需传完整的卡片字典

    try:
        response = requests.post(
            webhook_url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=10
        )
        response.raise_for_status()  # 触发HTTP错误
        result = response.json()
        if result.get("code") == 0:
            logging.info("飞书消息推送成功")
            return True
        else:
            logging.error(f"飞书推送失败：{result}")
            return False
    except Exception as e:
        logging.error(f"飞书推送异常：{str(e)}")
        return False


# ---------- 汇总格式化函数（修改：添加中文名） ----------
def format_summary(summary_data, name_map: Dict[str, str]):
    """格式化汇总数据，包含股票中文名"""
    # 基础信息
    base_info = f"""【选股结果汇总】
📅 交易日：{summary_data['trade_date']}
📊 策略总数：{summary_data['total_strategies']}
✅ 激活运行：{summary_data['active_strategies']}
❌ 运行失败：{len(summary_data['failed_strategies'])}"""

    # 失败策略信息
    failed_info = ""
    if summary_data["failed_strategies"]:
        failed_info = f"\n\n❌ 失败策略列表：\n" + "\n".join([f"- {err}" for err in summary_data["failed_strategies"]])

    # 各策略选股结果（匹配中文名）
    strategy_details = "\n\n📈 各策略选股结果："
    for res in summary_data["selector_results"]:
        # 给每个股票代码匹配中文名
        stocks_with_name = [f"{code}（{get_stock_name(code, name_map)}）" for code in res["stocks"]]
        if res["stocks"] == ["无符合条件股票"]:
            stocks_with_name = ["无符合条件股票"]
        stocks_str = ", ".join(stocks_with_name)
        strategy_details += f"\n\n【{res['alias']}】\n符合条件数：{res['stock_count']}\n股票列表：{stocks_str}"

    full_content = base_info + failed_info + strategy_details
    return full_content


if __name__ == "__main__":
    main()
