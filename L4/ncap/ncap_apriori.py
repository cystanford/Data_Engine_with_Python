import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori

# 数据加载
df = pd.read_csv('./ncap.csv', encoding='gbk') 
print(df.shape)
df.index = df['车型']
df.drop(['车型'], axis=1, inplace=True)
print(df)


# 数值>=12为TRUE，数值<12为FALSE
f = lambda x: 1 if x>=12 else 0
df = df.applymap(f)
print(df)
df.to_csv('temp.csv')
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
frequent_itemsets = frequent_itemsets.sort_values(by="support" , ascending=False) 
print("频繁项集：", frequent_itemsets)
# 显示全部的列
pd.options.display.max_columns=100
#rules = rules.sort_values(by="lift" , ascending=False) 
rules = rules.sort_values(by="confidence" , ascending=False) 
#print("关联规则：", rules[ (rules['lift'] >= 1) & (rules['confidence'] >= 0.5) ])
print('关联规则：', rules)
