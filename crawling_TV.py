import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
import time

chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)

driver.get('https://m.kinolights.com/theme/110')

time.sleep(3)
theme_list = [110]

# get TV_ids from main homepage
#source = requests.get("https://m.kinolights.com/theme/110")
#source = source.text

source = driver.page_source
print(source)
soup = BeautifulSoup(source, "html.parser")

driver.quit()

TVs = soup.find(id="listArea")
# print(TVs[0].prettify())
item_wrap = TVs.find_all("div", {"class": "movie-list-item-wrap"})[-1]
movie_list = item_wrap.find_all("div", {"class": "item-wrap"})
for i in movie_list:
    print(i.find("a"))

TV_labels = []
for i, tv in enumerate(TVs):
    if tv.find("div", {'class': 'type-label'}) != None:
        TV_labels.append(True)
    else:
        TV_labels.append(False)
TVs_id = [tv.select("a[href]")[0]["href"] for i, tv in enumerate(TVs) if TV_labels[i]]
print(TVs_id)

total_reviews = []
total_score = []

# get reviews ids from TV page
for i in TVs_id:
    print(i)
    tv_source = requests.get("https://m.kinolights.com" + i).text
    soup = BeautifulSoup(tv_source, "html.parser")
    reviews = soup.find_all("a", {'class': 'review-content-link'})
    Reviews_ids = list(set([review["href"] for review in reviews]))
    print(Reviews_ids)

    # get reviews and score in reviews page
    for j in Reviews_ids:
        review_source = requests.get("https://m.kinolights.com" + j).text
        soup = BeautifulSoup(review_source, "html.parser")
        one_review_rank = soup.select(".user-star-score")[0].text.replace('\n', '').replace('\t', '').replace('  ', '')
        one_review_title = soup.select("h3")[0].text.replace('\n', '').replace('\t', '').replace('  ', '')
        one_review_content = soup.find_all("div", {'class': 'contents'})[0].select("p")[0].text.replace('\n',
                                                                                                        '').replace(
            '\t', '').replace('  ', '')
        one_review = one_review_title + " " + one_review_content
        total_score.append(one_review_rank)
        print(one_review_rank)
        total_reviews.append(one_review)
        print(one_review)

result = pd.DataFrame()

result["reviews"] = total_reviews
result["score"] = total_score

result.to_excel("kinolights_Reviews.xlsx", encoding="utf8")


