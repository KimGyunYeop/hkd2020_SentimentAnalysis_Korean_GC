import datetime as datetime
from selenium.webdriver.support import expected_conditions as EC

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from dateutil.parser import parse
import csv
import datetime

from selenium.webdriver.support.wait import WebDriverWait

chromeDriver="C:\\Users\\sunny\\Downloads\\chromedriver_win32 (2)\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)

driver.get("https://www.laftel.net/finder")
time.sleep(5)

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(5)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

time.sleep(1)

source = driver.page_source
soup = BeautifulSoup(source, "html.parser")

tv = soup.find_all("div", {"class": "finder-card"})
#print(tv)
tvs = [id.select("a[href]")[0]["href"] for i, id in enumerate(tv)]
print(tvs)

href = pd.DataFrame()
href["url"] = tvs
href.to_csv("url.txt", encoding="utf8", sep="\t")

total_reviews = []
total_score = []

for tv in tvs:
    driver.get("https://www.laftel.net"+tv+"/review")
    #driver.get("https://www.laftel.net/item/14232/%EA%B3%A0%EC%96%91%EC%9D%B4%EC%9D%98-%EB%B3%B4%EC%9D%80/review")

    time.sleep(0.5)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(0.5)

    '''
    t=21
    while(1):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(0.5)


        if soup.find_all("div", {"class": "load-more"}):
            #print(soup.find_all("div", {"class": "load-more"})[0].text)
            #elem=driver.find_element_by_class_name('load-more')
            #elem=driver.find_element_by_css_selector("#root > div > div.item-template > div.item-panes > div.main > div:nth-child(1) > div.item-user-review-template > div > div.list > div.load-more")
            #print(elem.is_enabled())
            #elem.send_keys(Keys.ENTER)
            #driver.execute_script("arguments[0].click();", elem)

            #webdriver.ActionChains(driver).move_to_element(elem).send_keys('\n').perform()
            #time.sleep(2)
            #elem=driver.find_element_by_xpath("//*[@id=\"root\"]/div/div[2]/div[3]/div[1]/div[1]/div[3]/div/div[2]/div["+str(t)+"]")
            #elem.click()
            #driver.execute_script("arguments[0].click();", elem)

            time.sleep(2)
            print("===================================")
            t=t+20
        else:
            break
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")
    '''
    try:
        reviews = soup.find_all("div", {"class": "list"})[0].find_all("div", {"class": "text"})
        print(len(reviews))
        scores = soup.find_all("div", {"class": "list"})[0].find_all("div", {"class": "stars"})
        print(len(scores))
    except:
        continue


    for review,score in zip(reviews,scores):
        one_review=review.text.replace('\n', ' ').replace('\t', '').replace('  ', '')
        temp=score.find_all("svg")
        stars = [tem.select("path[fill]")[0]["fill"] for i, tem in enumerate(temp)]

        count=0
        for star in stars:
            if star=='#FFDA00':
                count=count+1

        if count==5:
            total_score.append(1)
        elif count<4:
            total_score.append(0)
        else:
            continue

        print(one_review)
        print(count)
        total_reviews.append(one_review)

print(len(total_reviews))
print(len(total_score))

driver.quit()

result = pd.DataFrame()
result["reviews"] = total_reviews
result["label"] = total_score

for i in range(len(total_reviews)): #코멘트가 공백인 row를 drop
    if result["reviews"][i]=='' or result["reviews"][i]==' ':
        result=result.drop(i,axis=0)

result.to_csv("laftel_Reviews.txt", encoding="utf8", sep="\t")

