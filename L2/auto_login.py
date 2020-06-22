# 自动GitHub 提交表单
import time
from selenium import webdriver

# 需要将chromedriver放到Chrome\Application目录下
driver = webdriver.Chrome()

request_url = 'https://github.com/login'
driver.get(request_url)

driver.find_element_by_id('login_field').send_keys('test@wucai.com')
driver.find_element_by_id('password').send_keys('123cylearn!!')
driver.find_element_by_class_name('btn-block').click()
