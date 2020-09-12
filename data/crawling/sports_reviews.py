import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import matplotlib.pyplot as plt

chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
news_list = ['https://sports.news.naver.com/wfootball/news/index.nhn?isphoto=N&type=comment&date=20200806',
             'https://sports.news.naver.com/wfootball/news/index.nhn?isphoto=N&type=comment&date=20200805',
             'https://sports.news.naver.com/wfootball/news/index.nhn?isphoto=N&type=comment&date=20200803',
             'https://sports.news.naver.com/wfootball/news/index.nhn?isphoto=N&type=comment&date=20200802',
             'https://sports.news.naver.com/wfootball/news/index.nhn?isphoto=N&type=comment&date=20200801',
             'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=comment&date=20200806',
             'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=comment&date=20200805',
             'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=comment&date=20200804',
             'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=comment&date=20200803',
             'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=comment&date=20200802',
             'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=comment&date=20200801',
             'https://sports.news.naver.com/volleyball/news/index.nhn?isphoto=N&type=comment&date=20200806',
             'https://sports.news.naver.com/volleyball/news/index.nhn?isphoto=N&type=comment&date=20200805',
             'https://sports.news.naver.com/basketball/news/index.nhn?isphoto=N&type=comment&date=20200806'
             ]
total_reviews = []
total_score = []
total_tmp = []
for i in range(len(news_list)):
    driver = webdriver.Chrome(chromeDriver)
    driver.get(news_list[i])
    print(i + 1, "/", len(news_list))
    time.sleep(3)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")
    news_soup = soup.find("div", {"class": "news_list"})
    news = news_soup.find_all("a", {"class": "thmb"})
    news_id = [new["href"] for i, new in enumerate(news)]


    driver.quit()
    # get reviews ids from TV page
    for j, id in enumerate(news_id):
        print(j + 1,"/",len(news_id))
        try:
            driver = webdriver.Chrome(chromeDriver)
            driver.get("https://sports.news.naver.com"+id)
            time.sleep(5)
            review_source = driver.page_source
            soup = BeautifulSoup(review_source, "html.parser")
            reaction = soup.find("span", {'style': 'z-index: 3;'})["class"][1]
            if reaction == "__reaction__angry" or reaction == "__reaction__sad":
                label = 0
            elif reaction == "__reaction__like" or reaction == "__reaction__fan":
                label = 1
            else:
                continue
            driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/div[2]/div[1]/a').click()
            time.sleep(3)
            review_source = driver.page_source
            soup = BeautifulSoup(review_source, "html.parser")
        except:
            continue
        driver.quit()
        reviews = soup.find_all("span", {'data-lang': 'ko'})
        reviews = [review.text.replace('\n', '').replace('\t', '').replace('\r', '').replace('  ', '') for review in reviews]
        labels = [label for _ in range(len(reviews))]

        total_score.extend(labels)
        total_reviews.extend(reviews)
        total_tmp.extend([reaction for _ in range(len(reviews))])

result = pd.DataFrame()
result["reviews"] = total_reviews
result["label"] = total_score
print(len(result))
print(len(result[result["label"]==1]))
print(len(result[result["label"]==0]))
result.to_csv("sports_Reviews.txt", encoding="utf8", sep="\t")

