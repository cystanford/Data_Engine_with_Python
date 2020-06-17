# 对汽车投诉信息进行数据预处理
import pandas as pd

# 分析type的内容，拆分成 年份, 发动机, 变速器（手动/自动），其他
def analyze(type):
	# 一般第一个为年份
	#year = type[0]
	year, engine, transmission, other = '', '', '', ''
	engine_list = ['1.2T', '1.4T', '1.4L', '1.4TSI', '1.5L', '1.5T', '1.5TD', '1.6L', '1.6T', '1.6THP', '1.8L', '1.8T', '1.8TD', '1.8TSI', '2.0L', '2.0T', '2.4L', '2.5L', '2.5T', '14T', '20T', '30T', '230TSI', '350T', '280TSI', '260T', '300T', '300TGI', '330TSI', '350THP', '350T', 'TSI280', '400TGI']
	for i in type:
		# 如果最后一个字为款，为年份
		if type.index(i)==0 and i[-1:] == '款':
			year = i[:-1]
			continue
		if i == '手动' or i == '自动':
			transmission = i
			continue
		if i in engine_list: 
			engine = i
			continue
		other = other + ' ' + i
	return year, engine, transmission, other


# 数据加载
result = pd.read_csv('car_complain.csv')
result['type_year'] = ''
result['type_engine'] = ''
result['type_transmission'] = ''
result['type_other'] = ''

# 分析type字段，拆分多个字段
for i, row in result.iterrows():
	year, engine, transmission, other = analyze(row['type'].split(' '))
	result.loc[i, 'type_year'] = year
	result.loc[i, 'type_engine'] = engine
	result.loc[i, 'type_transmission'] = transmission
	result.loc[i, 'type_other'] = other
	#print(year, engine, transmission, other)
# 删除列
result = result.drop(columns=['type'], axis=1)
result.to_csv('car_complain_data_clean.csv', index=False)
