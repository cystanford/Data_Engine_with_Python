#!/usr/bin/env python
# coding: utf-8

# In[66]:


# 使用Prophet预测manning未来365天的页面流量
# 从2007年12月10日开始
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 读入数据集
df = pd.read_csv('./manning.csv')
#print(df.head())
#print(df.tail())
# 拟合模型
model = Prophet()
print(model.growth)
#model = Prophet(growth='logistic', seasonality_mode='multiplicative')
model.fit(df)
#print(model.changepoints)
#print(len(model.changepoints))
# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = model.make_future_dataframe(periods=365)
#print(future.tail())

# 预测数据集
forecast = model.predict(future)
#print(forecast.columns)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 展示预测结果
model.plot(forecast)
plt.show()


# In[63]:


#print(forecast)
forecast.tail()
#print(forecast.shape)


# In[67]:


# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
model.plot_components(forecast)
print(forecast.columns)
#print(model.changepoints)


# In[49]:


# 饱和增长
m = Prophet(growth='logistic')
df['cap'] = 8.5
m.fit(df)

# 预测未来 5 年的数据
future = m.make_future_dataframe(periods=1826)
print(future)
# 将未来的承载能力设定的和历史数据一样，即8.5
future['cap'] = 8.5
fcst = m.predict(future)
fig = m.plot(fcst)
print(future)


# In[50]:


# 预测饱和减少, 修改y值
df['y'] = 10 - df['y']
df['cap'] = 6
# 设置下限
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)


# In[68]:


from fbprophet.plot import add_changepoints_to_plot
fig = model.plot(forecast)
# plt.gca()获得当前的Axes对象ax
# 获取显著的突变点的位置

a = add_changepoints_to_plot(fig.gca(), model, forecast)
#print(model.changepoints)
#print(len(model.changepoints))
#print(model.seasonality_mode)
#print(model.changepoint_range)
#print(model.growth)


# In[7]:


# 指定突变点的位置
m = Prophet(changepoints=['2014-01-01'])
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
m.plot(forecast);


# In[71]:


# 对节假日建模
# 将节日看成是一个正态分布，把活动期间当做波峰，lower_window 以及upper_window 的窗口作为扩散
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
print(holidays)


# In[78]:


m = Prophet(holidays=holidays)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
#print(forecast.tail())
# 通过 forecast 数据框，展示节假日效应
print(forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][['ds', 'playoff', 'superbowl']][-10:])


# In[79]:


# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应,holidays项
m.plot_components(forecast)
print(forecast.columns)

