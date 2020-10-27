import pandas as pd
import csv
for i in range(1,178):  # 爬取全部页
    tb = pd.read_html('http://s.askci.com/stock/a/?reportTime=2017-12-31&pageNum=%s' % (str(i)))[3]
    tb.to_csv(r'1.csv', mode='a', encoding='utf_8_sig', header=1, index=0)