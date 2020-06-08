# 对汽车投诉信息进行分析
import pandas as pd

result = pd.read_csv('car_complain.csv')
#print(result)
# 将genres进行one-hot编码（离散特征有多少取值，就用多少维来表示这个特征）
result = result.drop('problem', 1).join(result.problem.str.get_dummies(','))
print(result.columns)
tags = result.columns[7:]
#print(tags)

df= result.groupby(['brand'])['id'].agg(['count'])
df2= result.groupby(['brand'])[tags].agg(['sum'])
df2 = df.merge(df2, left_index=True, right_index=True, how='left')
# 通过reset_index将DataFrameGroupBy => DataFrame
df2.reset_index(inplace=True)
#df2.to_csv('temp.csv')
df2= df2.sort_values('count', ascending=False)
print(df2)
#print(df2.columns)
#df2.to_csv('temp.csv', index=False)
query = ('A11', 'sum')
print(df2.sort_values(query, ascending=False))
