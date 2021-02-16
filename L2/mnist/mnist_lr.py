# -*- coding: utf-8 -*-
# 使用LR进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.title('Hand Written Digits')
plt.imshow(digits.images[0])
plt.show()
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
#train_ss_x = ss.fit_transform(train_x)
#test_ss_x = ss.transform(test_x)
data = ss.fit_transform(data)

# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# 创建LR分类器
lr = LogisticRegression()
# fit代表拟合，训练网络模型，让参数自动拟合
lr.fit(train_x, train_y)
predict_y=lr.predict(test_x)
print('LR准确率: %0.4lf' % accuracy_score(predict_y, test_y))
