import numpy as np


def run_backtest(data_dict, selector, hold_days=3):
    """
    简易回测引擎
    """
    results = []
    for code, df in data_dict.items():
        # 运行你的 Selector 选股
        signals = selector.analyze(df)  # 假设这是你的选股接口

        for signal_date in signals:
            buy_price = df.loc[df['date'] == signal_date, 'close'].values[0]
            # 计算 hold_days 后的卖出价
            sell_idx = df.index[df['date'] == signal_date] + hold_days

            if len(sell_idx) > 0 and sell_idx[0] < len(df):
                sell_price = df.loc[sell_idx[0], 'close']
                profit = (sell_price - buy_price) / buy_price
                results.append(profit)

    avg_profit = np.mean(results) * 100
    win_rate = len([r for r in results if r > 0]) / len(results) * 100
    print(f"回测结果: 平均收益 {avg_profit:.2f}%, 胜率 {win_rate:.2f}%")