import tushare as ts

# tushare版本 1.4.24
token = "26d6a5877ad3da85312145f9975b98873a5291b5a50e3af33d6b77671b70"

pro = ts.pro_api(token)

pro._DataApi__token = token  # 保证有这个代码，不然不可以获取
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'  # 保证有这个代码，不然不可以获取

def get_stock_name_map():
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
    # 转换为字典：{代码: 名称}
    name_map = dict(zip(df['ts_code'], df['name']))
    return name_map


