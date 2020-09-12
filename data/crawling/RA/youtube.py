import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import csv

chromeDriver="C:\\Users\\sunny\\Downloads\\chromedriver_win32 (2)\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)

driver.get("https://www.youtube.com/watch?v=mk0a_Zq7M3U")
time.sleep(3)

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

last_page_height = driver.execute_script("return document.documentElement.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(3.0)       # 인터발 1이상으로 줘야 데이터 취득가능(롤링시 데이터 로딩 시간 때문)
    new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height

source = driver.page_source
soup = BeautifulSoup(source, "lxml")

driver.quit()

youtube_comments = soup.select("yt-formatted-string#content-text")

str_youtube_comments = []  # USER 댓글 내용 배열
label =[]

# REPLACE DATA
for i in range(len(youtube_comments)):

    str_tmp = str(youtube_comments[i].text)
    str_tmp = str_tmp.replace('\n', '')
    str_tmp = str_tmp.replace('\t', '')
    str_tmp = str_tmp.replace('   ','')
    str_youtube_comments.append(str_tmp)
    label.append(0)

## MODIFY VIEW FORMAT
pd_data = {"review":str_youtube_comments,"label":label}
youtube_pd = pd.DataFrame(pd_data)

youtube_pd.to_csv("youtube_comments_positive.txt", encoding="utf8",sep='\t')

