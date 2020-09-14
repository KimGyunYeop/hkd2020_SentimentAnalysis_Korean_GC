import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.keys import Keys
import time

chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)

total_content_pos = []
total_content_neg = []
total_score = []

#############################################################################################
# 페이지 이동하기
k = 0
while True:
    year = 2020 - k
    if year < 2004:
        break

    driver.get("https://movie.daum.net/boxoffice/yearly?year=" + str(year))
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

    time.sleep(3)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")

    #############################################################################################
    #한 페이지에서 게시물들 url 가져오기

    movie = []
    for href in soup.find_all("a", class_="desc_movie"):
        if len(movie) == 15:
            break
        temp = href.attrs['href']
        temp = temp.replace("/moviedb/main?movieId=", "")
        movie.append(temp)

    #print(movie)

    for i in movie:
        for page in range(1, 200):
            try:
                driver.get("https://movie.daum.net/moviedb/grade?movieId=" + i + "&type=netizen&page=" + str(page))
            except:
                continue

            time.sleep(0.5)

            tv_source = driver.page_source
            soup = BeautifulSoup(tv_source, "html.parser")

            rate = []
            rate = soup.find_all("em", {'class': 'emph_grade'})
            for r in range(len(rate)):
                rate[r] = int(rate[r].text)

            #print(rate)

            content = soup.find_all("p", {'class': 'desc_review'})
            for c in range(len(content)):
                content[c] = content[c].text
                content[c] = content[c].replace("   ", "").replace("\n", "")
                if len(content[c]) > 2:
                    content[c] = content[c][45:]
                    content[c] = content[c][:len(content[c])-40]
                elif content[c] == " ":
                    content[c] = ""

            #print(content)

            for n in range(len(rate)):
                if content[n] == "" or 4 < rate[n] < 8:
                    continue
                if rate[n] >= 8:
                    total_score.append(1)
                    total_content_pos.append(content[n])
                elif rate[n] <= 4:
                    total_score.append(0)
                    total_content_neg.append(content[n])

            time.sleep(0.1)

    if len(total_content_neg) > 13000:
        break
    k += 1

driver.quit()

#############################################################################################
# 저장
total_content = total_content_pos + total_content_neg
result = pd.DataFrame()
result["label"] = total_score
result["reviews"] = total_content

result.to_csv("daumMovie_Reviews_v2_12.txt", encoding="utf8", sep="\t")