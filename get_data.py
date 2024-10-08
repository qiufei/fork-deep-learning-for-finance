import tushare as ts
ts.set_token('ba1646815a79a63470552889a69f957f5544bef01d3f082159bf8474')

# 初始化pro接口
pro = ts.pro_api()

df = pro.query('daily', ts_code='000001.SZ', start_date='20180701', end_date='20180718')

# 选取trade_date和close列
# trade_date是交易日期，close是收盘价
close = df['trade_date','close']

'''
The next step is to import and transform the close price data. Remember, you are trying to forecast daily returns, which means that you must select only the close column
and then apply a differencing function on it so that prices become differenced.
'''