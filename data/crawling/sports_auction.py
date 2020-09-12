import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

chromeDriver = "C:\\Users\\parksoyoung\\Downloads\\chromedriver_win32\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
driver.get('https://mobileticket.interpark.com/Goods/GoodsInfo/info?GoodsCode=20001874&is1=ticket&is2=product')
time.sleep(3)

driver.find_element_by_xpath('//*[@id="root"]/div[@class="contents"]/div[@class="productsInformation"]/div[@class="productsTabWrap"]'
                             '/*[@id="productsTab"]/ul/li[3]').click()
elem = driver.find_element_by_tag_name("body")

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    for i in range(10):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(30)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

time.sleep(3)
source = driver.page_source
soup = BeautifulSoup(source, "html.parser")

driver.quit()

sports = soup.find("ul", {"id": "writerInfo"})
comments_li =sports.find_all("li")
print(len(comments_li))

result = pd.DataFrame()
titles = []
reviews = []
labels = []
rates = []

for li in comments_li:
    #print(li.find("div", {"class": "userBoardTitle"}).find("b").find(text=True))
    title = li.find("div", {"class": "userBoardTitle"}).find("b").find(text=True)
    #print(li.find("div", {"class": "boardContentTxt"}).find(text=True))
    text = li.find("div", {"class": "boardContentTxt"}).find(text=True)
    #print(li.find("div", {"class": "shareInfo"}).find("div").get("class"))
    rate = li.find("div", {"class": "shareInfo"}).find("div").get("class")
    score = int(rate[1][5:])
    #print(score)
    if score >= 8:
        label = 1
    elif score <= 6:
        label = 0
    else:
        continue

    titles.append(title)
    reviews.append(text)
    labels.append(label)
    rates.append(score)



result['title'] = titles
result['review'] = reviews
result['label'] = labels
result['rating'] = rates

result.to_csv('sports_auc.txt', encoding="utf8", sep="\t")