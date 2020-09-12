import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

chromeDriver = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)
driver.get('https://m.kinolights.com/what-to-watch')
time.sleep(3)
#element = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.CLASS_NAME, "media-type-btn.active")))
#ActionChains(driver).move_to_element(element).click().perform()
driver.find_element_by_xpath('//*[@id="contents"]/div[5]/div/div/div/button').click()
driver.find_element_by_xpath('//*[@id="contents"]/div[2]/div[2]/button[3]').click()
time.sleep(3)
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

source = driver.page_source
soup = BeautifulSoup(source, "html.parser")

driver.quit()

TVs = soup.find_all("div", {"class": "item-wrap"})
print(len(TVs))
TVs_id = []
TV_labels = []

for i, tv in enumerate(TVs):
    if tv.find("div", {'class' : 'type-label'}) != None:
        TV_labels.append(True)
    else:
        TV_labels.append(False)
TVs_id = [tv.select("a[href]")[0]["href"] for i, tv in enumerate(TVs) if tv.find("span", {'class' : 'score'}).text != "-"]
print(len(TVs_id))

total_ids = []
total_reviews = []
total_score = []

# get reviews ids from TV page
for i in TVs_id:
    print(i)
    driver = webdriver.Chrome(chromeDriver)
    driver.get("https://m.kinolights.com" + i+"/reviews")

    time.sleep(3)
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(0.8)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    tv_source = driver.page_source
    #tv_source = requests.get("https://m.kinolights.com" + i+"/reviews").text
    soup = BeautifulSoup(tv_source, "html.parser")
    driver.quit()
    reviews = soup.find_all("a", {'class': 'review-content-link'})
    Reviews_ids = list(set([review["href"] for review in reviews]))
    print(Reviews_ids)

    # get reviews and score in reviews page
    for j in Reviews_ids:
        review_source = requests.get("https://m.kinolights.com" + j).text
        soup = BeautifulSoup(review_source, "html.parser")
        try:
            one_review_rank = soup.select(".user-star-score")[0].text.replace('\n', '').replace('\t', '').replace('  ', '')
        except:
            continue
        if one_review_rank == "-":
            continue
        else:
            one_review_rank = float(one_review_rank)
        one_review_title = soup.select("h3")[0].text
        if one_review_title !="":
            one_review_title = one_review_title.replace('\n', '').replace('\t', '').replace('  ', '')
        one_review_content = soup.find_all("div", {'class': 'contents'})[0].select("p")[0].text
        if one_review_content != "":
            one_review_content = one_review_content.replace('\n', '').replace('\t', '').replace('  ', '')

        one_review = one_review_title + " " + one_review_content
        if one_review_rank > 4:
            total_score.append(1)
        elif one_review_rank < 2:
            total_score.append(0)
        else :
            continue
        print(one_review_rank)
        total_reviews.append(one_review)
        print(one_review)
        total_ids.append(i[7:])
result = pd.DataFrame()
result["id"] = total_ids
result["reviews"] = total_reviews
result["label"] = total_score

result.to_csv("kinolights_Reviews.txt", encoding="utf8", sep="\t")

