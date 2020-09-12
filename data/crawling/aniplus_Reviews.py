import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

chromeDriver="C:\\Users\\sunny\\Downloads\\chromedriver_win32 (2)\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
driver.get('http://www.aniplustv.com/#/tv/default.asp?gCode=TV&sCode=001')
time.sleep(0.5)
elem = driver.find_element_by_tag_name("body")

source = driver.page_source
soup = BeautifulSoup(source, "html.parser")

pages = soup.find("div", {'class': 'paging'}).find_all("span", {'class': 'num'})
#print(pages)

num = list()

count=0
check = 0

for page in pages:
    if check == 0:
        #print(page.find("span", {"class": "on"}))
        page_num = int(page.find("span", {"class": "on"}).text)
    else:
        page_num = int(page.find("a").text)
    num.append(page_num)
    check = 1
#print(num[-1])

movie_page = "http://www.aniplustv.com/#/tv/default_category_list.asp?gotoPage="
for i in range(1, num[-1] + 1):
    movie_page = "http://www.aniplustv.com/#/tv/default_category_list.asp?gotoPage="

    print(movie_page+str(i))

    driver.get(movie_page+str(i))
    time.sleep(3)

    tv_source = driver.page_source
    soup = BeautifulSoup(tv_source, "html.parser")

    #작품 목록 페이지 이동
    #######################################################################

    t = 0
    string = list()

    for href in soup.find("ul", {"class": "list"}).find_all("li"):
        movie_page = href.find("a")["href"]
        movie_page = movie_page.replace("program_view","30thtalkList_new")
        string.append(movie_page)
        t = t + 1

    total_ids = []
    total_reviews = []
    total_score = []
    #print(string)

    for j in range(len(string)):
        driver.get("http://www.aniplustv.com/" + string[j])
        #print("http://www.aniplustv.com/" + string[j])
        time.sleep(0.5)

        soup = BeautifulSoup(driver.page_source , "html.parser")

        try:
            review_page_list = soup.find("div",{"id":"divPage"}).find_all("span",{"class":"num"})
        except:
            continue

        if len(review_page_list) is 0:
            continue
        #print(review_page_list)
        time.sleep(3)

        for index in range(len(review_page_list)):
            reivew_link = string[j]+"&gotoPage="+str(index+1)

            driver.get("http://www.aniplustv.com/"+reivew_link)
            time.sleep(0.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")

            reviews = soup.find_all("td", {'class': 'vw_write'})
            scores = soup.find_all("td", {'class': 'wp_star'})

            for review, score in zip(reviews, scores):

                one_review = review.text.replace('\n', '').replace('\t', '').replace('  ', '')

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
                count=count+1

                # total_ids.append(i[16:])
            # print(len(total_ids))

driver.quit()

print(count)

result = pd.DataFrame()
result["reviews"] = total_reviews
result["label"] = total_score
result=result.drop_duplicates('reviews')

result.to_csv("aniplus_Reviews.txt", encoding="utf8", sep="\t")


