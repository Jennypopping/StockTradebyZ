import tushare as ts
import pandas as pd

# --- 配置测试环境 ---
TOKEN = "26d6a5877ad3da85312145f9975b98873a5291b5a50e3af33d6b77671b70"
PROXY_URL = 'http://lianghua.nanyangqiankun.top'
TARGET_DATE = "20260303"

# 初始化
ts.set_token(TOKEN)
pro = ts.pro_api(TOKEN)
# 强制指定代理地址
pro._DataApi__http_url = PROXY_URL

def test_proxy_data():
    print(f"🚀 正在通过代理测试 {TARGET_DATE} 的数据更新情况...")
    print(f"代理地址: {PROXY_URL}\n")

    # 1. 测试日线行情 (daily) - 选一只大票测试
    try:
        df_daily = pro.daily(ts_code='600519.SH', trade_date=TARGET_DATE)
        if not df_daily.empty:
            print(f"✅ 日线行情：已更新！")
            print(df_daily[['ts_code', 'trade_date', 'open', 'close', 'vol']])
        else:
            print(f"❌ 日线行情：未找到数据（返回为空）。")
    except Exception as e:
        print(f"💥 日线接口报错: {e}")

    print("-" * 30)

    # 2. 测试交易日历 (trade_cal) - 查看代理认为的最晚交易日
    try:
        # 查最近一两周的开盘记录
        df_cal = pro.trade_cal(exchange='SSE', is_open=1,
                               start_date='20260220',
                               end_date=TARGET_DATE)
        if not df_cal.empty:
            last_day = df_cal.iloc[-1]['cal_date']
            print(f"📅 代理日历显示最新交易日为: {last_day}")
            if last_day == TARGET_DATE:
                print("✅ 代理日历已同步到 3月2日。")
            else:
                print(f"⚠️ 代理日历落后，目前只到 {last_day}。")
        else:
            print("❌ 代理日历接口未返回任何开启的交易日。")
    except Exception as e:
        print(f"💥 日历接口报错: {e}")

if __name__ == "__main__":
    test_proxy_data()