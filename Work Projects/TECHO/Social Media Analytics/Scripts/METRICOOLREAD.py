import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

url = "https://app.metricool.com/evolution/facebookPage?blogId=2460321&userId=2083682"
driver.get(url)
time.sleep(3)

cookies = pickle.load(open("cookies.pkl", "rb"))
for cookie in cookies:
    cookie.pop('samesite', None)
    driver.add_cookie(cookie)

driver.get(url)

time.sleep(5)


input("Press Enter to close the browser...")
driver.quit()