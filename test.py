import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

import glob


def clear_csv_files(directory):
    # 定义需要保留的表头
    HEADER = "date,open,close,high,low,volume,amount,capital"
    # 构建匹配模式，找到 data 文件夹下所有的 csv 文件
    csv_pattern = os.path.join(directory, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"在目录 '{directory}' 中没有找到任何 CSV 文件。")
        return

    print(f"正在处理 {len(csv_files)} 个文件（保留表头，清空数据）...")

    for file_path in csv_files:
        try:
            # 先读取文件，检查并保留表头
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 打开文件，只写入表头（如果原有表头不一致，强制替换为标准表头）
            with open(file_path, 'w', encoding='utf-8') as f:
                # 优先使用文件原有表头（如果存在），否则使用标准表头
                if lines and lines[0].strip() != "":
                    # 验证原有表头是否正确，不正确则替换
                    original_header = lines[0].strip()
                    if original_header == HEADER:
                        f.write(original_header + '\n')
                    else:
                        f.write(HEADER + '\n')
                        print(f"⚠️ {os.path.basename(file_path)} 原有表头错误，已替换为标准表头")
                else:
                    f.write(HEADER + '\n')

            print(f"已处理: {os.path.basename(file_path)}（保留表头，清空数据）")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    print("--- 所有文件已处理完毕（表头保留，数据清空）---")

def cleanup_csv_files(data_dir="./data", target_date="2026-03-03"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 错误: 找不到目录 {data_dir}")
        return

    csv_files = list(data_path.glob("*.csv"))
    print(f"🔍 正在扫描 {len(csv_files)} 个文件...")

    modified_count = 0

    for file_path in tqdm(csv_files, desc="清理进度"):
        try:
            # 1. 加载数据
            df = pd.read_csv(file_path)

            if df.empty:
                continue

            # 2. 检查是否存在目标日期 (兼容字符串格式)
            mask = df['date'].astype(str).str.contains(target_date)

            if mask.any():
                # 3. 过滤掉目标日期，保留其他行
                df_cleaned = df[~mask]

                # 4. 回写覆盖 (保持原有的 4 位小数格式)
                # 使用 float_format='%.4f' 确保写回时补齐 0
                df_cleaned.to_csv(file_path, index=False, float_format='%.4f')
                modified_count += 1

        except Exception as e:
            print(f"⚠️ 处理文件 {file_path.name} 时出错: {e}")

    print(f"\n✨ 清理完成！")
    print(f"✅ 已从 {modified_count} 个文件中删除了 {target_date} 的数据。")


if __name__ == "__main__":
    # cleanup_csv_files()
    target_dir = "data"
    clear_csv_files(target_dir)