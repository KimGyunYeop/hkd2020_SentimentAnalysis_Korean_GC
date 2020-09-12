import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
driver.get('https://pedia.watcha.com/ko-KR')
time.sleep(3)
driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/header/nav/div/div/ul/li[3]/button').click()
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

driver.quit()

TVs = soup.find_all("li", {"class": "css-106b4k6-Self e3fgkal0"})
print(TVs)
TVs_id = []
TV_labels = []

TVs_id = [tv.select("a[href]")[0]["href"] for i, tv in enumerate(TVs)]
TVs_id = list(set(TVs_id))
print(len(TVs_id))

total_ids = []
total_reviews = []
total_score = []

# get reviews ids from TV page
for i in TVs_id:
    print(i)
    review_source = requests.get("https://pedia.watcha.com" + i).text
    soup = BeautifulSoup(review_source, "html.parser")

    try:
        number = int(soup.find("span", {'class': 'css-1odj31c-TitleSuffixForNumber eupnho10'}).text[:-1])
    except:
        number = int(soup.find("span", {'class': 'css-1odj31c-TitleSuffixForNumber eupnho10'}).text)
    driver = webdriver.Chrome(chromeDriver)
    driver.get("https://pedia.watcha.com" + i + "/comments")

    time.sleep(3)
    print(number)

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
    tv_source = driver.page_source
    soup = BeautifulSoup(tv_source, "html.parser")
    driver.quit()
    reviews = soup.find_all("div", {'class': 'css-aintwb-Text e1xxz10x0'})
    scores = soup.find_all("div", {'class': 'css-1a97064-UserActionStatus e1oskw6f4'})
    print(scores)
    for review, score in zip(reviews, scores):
        # get reviews and score in reviews page
        one_review = review.select("span")[0].text.replace('\n', '').replace('\t', '').replace('  ', '')
        try:
            one_review_rank = float(score.select("span")[0].text.replace('\n', '').replace('\t', '').replace('  ', ''))
        except:
            continue

        if one_review_rank > 4:
            total_score.append(1)
        elif one_review_rank < 2:
            total_score.append(0)
        else :
            continue
        print(one_review_rank)
        total_reviews.append(one_review)
        print(one_review)
        total_ids.append(i[16:])
    print(len(total_ids))

result = pd.DataFrame()
result["id"] = total_ids
result["reviews"] = total_reviews
result["label"] = total_score

result.to_csv("kinolights_Reviews.txt", encoding="utf8", sep="\t")

