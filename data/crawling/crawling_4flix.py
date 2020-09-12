import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import matplotlib.pyplot as plt

chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
driver.get('https://www.4flix.co.kr/board/netflix')
driver.find_element_by_xpath('/html/body/div[2]/div[1]/div[2]/div[3]/ul/li[2]/a').click()
time.sleep(1)

elem_login = driver.find_element_by_id("login_id")
elem_login.clear()
elem_login.send_keys("ckatodmlrna")

elem_login = driver.find_element_by_id("login_pw")
elem_login.clear()
elem_login.send_keys("chj2338860^")

driver.find_element_by_xpath('//*[@id="login_fs"]/input').click()

time.sleep(1)
driver.find_element_by_xpath('//*[@id="bo_cate_ul"]/li[3]').click()
list_id_10 = [1,3,4,5,6,7,8,9,10,11]
list_id_20 = [3,4,5,6,7,8,9,10,11,12]
total_TVs_id = []
i=0
while i<25:
    time.sleep(1)
    source = driver.page_source
    soup = BeautifulSoup(source, "html.parser")

    TVs = soup.find_all("li", {"class": "gall_li"})
    TVs_id = []
    TV_labels = []

    TVs_id = [tv.select("a[href]")[0]["href"] for i, tv in enumerate(TVs)]
    total_TVs_id.extend(TVs_id)
    j = i % 10
    if i < 10:
        list = list_id_10
    else:
        list = list_id_20
    try:
        driver.find_element_by_xpath('/html/body/div[3]/div[2]/div[2]/div/nav/span/a['+str(list[j])+']').click()
    except:
        print("error")
        break
    i+=1
print(i)
print(total_TVs_id)
driver.quit()
print(len(total_TVs_id))

total_reviews = []
total_score = []
total_tmp = []

# get reviews ids from TV page
for i, id in enumerate(total_TVs_id):
    print(i,"/",len(total_TVs_id))
    review_source = requests.get(id).text
    soup = BeautifulSoup(review_source, "html.parser")

    review_soups = soup.select("article[id]")
    tv_scores = []
    tv_reviews = []
    for review_soup in review_soups:
        try:
            score = review_soup.find("span", {'class': 'star_score_span'})
            review = review_soup.find("div", {'class': 'c_content'})
            one_review = review.text.replace('\n', '').replace('\t', '').replace('\r', '').replace('  ', '')
            one_score = int(score["style"][6:-1])
            total_tmp.append(one_score)
        except:
            continue
        if one_score >= 80:
            one_label = 1
        elif one_score <= 40:
            one_label = 0
        else :
            continue
        tv_scores.append(one_label)
        tv_reviews.append(one_review)

    print(tv_scores)
    print(tv_reviews)

    total_score.extend(tv_scores)
    total_reviews.extend(tv_reviews)

result = pd.DataFrame()
result["reviews"] = total_reviews
result["label"] = total_score
print(len(total_tmp))
plt.hist(total_tmp)
print(len(result))
print(len(result[result["label"]==1]))
print(len(result[result["label"]==0]))
result.to_csv("4flix_Reviews.txt", encoding="utf8", sep="\t")
plt.show()

