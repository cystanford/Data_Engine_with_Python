import requests
from bs4 import BeautifulSoup
# 请求URL
url = 'http://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-1.shtml'
# 得到页面的内容
headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
html=requests.get(url,headers=headers,timeout=10)
content = html.text
# 通过content创建BeautifulSoup对象
soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')

#输出第一个 title 标签
print(soup.title)
#输出第一个 title 标签的标签名称
print(soup.title.name)
#输出第一个 title 标签的包含内容
print(soup.title.string)
