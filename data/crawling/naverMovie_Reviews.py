import datetime as datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.keys import Keys
import time
from dateutil.parser import parse
import csv
import datetime

chromeDriver="C:\\Users\\sunny\\Downloads\\chromedriver_win32 (2)\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)

#############################################################################################
#영화 목록 페이지 이동하기

test=[101,201,301,401,501,601,701, 801, 901, 1001, 1101, 1201]
for i in range(12):
    test[i]=test[i]+2019*10000
print(test)

id = []
total =0

for t in test:
    driver.get("https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=cur&date="+str(t))
    time.sleep(3)

    elem = driver.find_element_by_tag_name("body")
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(0.5)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    time.sleep(1)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")

    #############################################################################################
    #한 페이지에서 영화 url들 가져오기

    movie = soup.find_all("td", {"class": "title"})
    temp = [mov.select("a[href]")[0]["href"] for i, mov in enumerate(movie)]
    total=total+len(temp)
    id=id+temp

id=list(set(id))
print("중복값O : "+str(total))
print("중복값X : "+str(len(id)))

total_reviews = []
total_score = []

for i in id:
    code = i.split('=')

    count=0
    while(1):
        count=count+1
        driver.get("https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=" + code[1]
                   + "&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page="
                   + str(count))
        time.sleep(1)
        source = driver.page_source
        soup = BeautifulSoup(source, "html.parser")

        if soup.find_all("a", {"class": "pg_next"}):
            print(str(count)+"---------------------------------------------------------------------")
        else:
            break

        reviews = soup.find_all("div", {"class": "score_reple"})
        scores = soup.find_all("div", {"class": "star_score"})

        for review, score in zip(reviews, scores):
            one_review = review.select("span")[-2].text.replace('\n', ' ').replace('\t', '').replace('  ', '').replace(
                '\"', '')
            try:
                one_review_rank = float(score.select("em")[0].text)
            except:
                continue

            if one_review_rank >= 8:
                total_score.append(1)
            elif one_review_rank <= 4:
                total_score.append(0)
            else:
                continue

            print(one_review_rank)
            total_reviews.append(one_review)
            print(one_review)

print(len(total_reviews))
print(len(total_score))

driver.quit()

result = pd.DataFrame()
result["reviews"] = total_reviews
result["label"] = total_score

for i in range(len(total_reviews)): #코멘트가 공백인 row를 drop
    if result["reviews"][i]=='' or result["reviews"][i]==' ':
        result=result.drop(i,axis=0)

result.to_csv("naverMovie_Reviews_2019.txt", encoding="utf8", sep="\t")