# 柱状图绘制
import pyecharts.options as opts
from pyecharts.charts import Bar

# 绘制单列柱状图
def work1():
	country = ["美国","巴西","俄罗斯", "西班牙", "英国","意大利","法国","德国","土耳其","伊朗"]
	data = [1666828, 347398, 335882, 281904, 258504, 229327, 182036, 179986, 155686, 133521]
	bar = (    
		Bar()    
		.add_xaxis(country)    
		.add_yaxis("累计确诊", data)
		.set_series_opts(label_opts=opts.LabelOpts(is_show=False))    
		.set_global_opts(title_opts=opts.TitleOpts(title="累计确诊国家 Top10"))
		)
	bar.render('temp.html')
	#bar.render_notebook()

# 绘制多列柱状图
def work2():
	# 绘制多列柱状图
	country = ["美国","西班牙","意大利","法国","德国","伊朗","英国","以色列","荷兰","奥地利"]
	data1 = [1666828, 347398, 335882, 281904, 258504, 229327, 182036, 179986, 155686, 133521]
	data2 = [21929, 16508, 9434, 1787, 2960, 669, 18, 199, 1186, 1869]    
	bar = (    
		Bar()    
		.add_xaxis(country)    
		.add_yaxis("累计确诊", data1)
		.add_yaxis("新增确诊", data2)
		.set_series_opts(label_opts=opts.LabelOpts(is_show=False))    
		.set_global_opts(title_opts=opts.TitleOpts(title="累计确诊国家 Top10"))
		)
	bar.render('temp.html')

work1()
#work2()


