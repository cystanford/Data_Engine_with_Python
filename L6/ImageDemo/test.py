# 使用部署好的API接口完成预测
import json
import urllib.request
import base64

with open("./dog.jpg", 'rb') as f:
    base64_data = base64.b64encode(f.read())
    s = base64_data.decode()

url = 'http://service-gxmg2oed-1255932437.sh.apigw.tencentcs.com/release/image'

print(urllib.request.urlopen(urllib.request.Request(
    url = url,
    data= json.dumps({'picture': s}).encode("utf-8")
)).read().decode("utf-8"))