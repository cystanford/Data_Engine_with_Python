import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori

# header=None，不将第一行作为head
dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None) 
# shape为(7501,20)
print(dataset.shape)

# 将数据存放到transactions中
transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0, 20):
        if str(dataset.values[i, j]) != 'nan':
           temp.append(str(dataset.values[i, j]))
    transactions.append(temp)
#print(transactions)
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.4)
print("频繁项集：", itemsets)
print("关联规则：", rules)
