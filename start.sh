#!/bin/bash

# --- 1. 环境与路径配置 ---
# 激活虚拟环境 (假设你的虚拟环境文件夹叫 venv)
source venv/bin/activate

# 定义全局汇总日志
START_LOG="start.log"

echo "------------------------------------------------" >> $START_LOG
echo "🚀 任务启动: $(date '+%Y-%m-%d %H:%M:%S')" >> $START_LOG

# --- 2. 步骤一：增量下载数据 ---
echo "📥 步骤 1: 开始增量获取 Tushare 数据..." >> $START_LOG

# 执行获取行情代码 (已整合你的参数)
python3 fetch_kline_new.py \
    --start 20250101 \
    --end today \
    --stocklist ./stocklist.csv \
    --exclude-boards star bj \
    --out ./data \
    --workers 4 >> $START_LOG 2>&1

# 检查退出码 ($? 为 0 代表成功)
if [ $? -eq 0 ]; then
    echo "✅ 步骤 1 成功：数据已更新或无需更新。" >> $START_LOG
else
    echo "❌ 步骤 1 失败：抓取过程出错，请检查 start.log 或 fetch.log" >> $START_LOG
    exit 1
fi

# --- 3. 步骤二：筛选股票并推送飞书 ---
echo "🔍 步骤 2: 开始执行股票筛选逻辑..." >> $START_LOG

# 执行选股代码 (已整合你的参数)
python3 select_stock.py \
    --data-dir ./data \
    --config ./configs.json >> $START_LOG 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 步骤 2 成功：选股完成并已推送到飞书。" >> $START_LOG
else
    echo "❌ 步骤 2 失败：筛选或推送过程出错，请检查 start.log" >> $START_LOG
    exit 1
fi

echo "🏁 任务结束: $(date '+%Y-%m-%d %H:%M:%S')" >> $START_LOG
echo "------------------------------------------------" >> $START_LOG