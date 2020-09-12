import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.keys import Keys
import time
from dateutil.parser import parse

chromeDriver="C://Users//ehrqo//Downloads//chromedriver_win32//chromedriver.exe"
driver = webdriver.Chrome(chromeDriver)


win_dates = []
win_times = []
win_wins = []
win_when=[]

team="NC" #경기결과 알고싶은 팀 넣는 변수

#############################################################################################
# 달별 페이지 열기

for month in range(4,10): #일단 이번년도 프로야구 개막~현재 까지의 경기결과
    driver.get("https://sports.news.naver.com/kbaseball/schedule/index.nhn?month=0" + str(month) + "&year=2020")
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
    # 해당 팀 위치 파악

    left_list = []  # 해당 팀이 몇번째 줄에서 왼쪽팀 인가
    right_list = []  # 해당 팀이 몇번째 줄에서 오른쪽팀 인가

    all_plays = soup.find_all("span", {"class", "team_lft"})
    count = 0
    for play in all_plays:
        if play.text == team:
            left_list.append(count)
        count = count + 1

    all_plays = soup.find_all("span", {"class", "team_rgt"})
    count = 0
    for play in all_plays:
        if play.text == team:
            right_list.append(count)
        count = count + 1

    #############################################################################################
    # 날짜, 시간 파악

    date = []
    times = []
    when=[]

    hours = soup.find_all("span", {"class", "td_hour"})
    for hour in hours:
        if hour.text!='-':
            times.append(hour.text.replace('\n', '').replace('\t', '').replace(' ', ''))

    if month==1 or month==3 or month==5 or month==7 or month==8 or month==10 or month==12:
        last_day=31
    else:
        last_day=30

    for i in range(0, last_day):
        n = soup.find_all("div", {"class", "tb_wrap"})[0].find_all("tbody")[i].find_all("td")[0]["rowspan"]
        d = soup.find_all("div", {"class", "tb_wrap"})[0].find_all("tbody")[i].find_all("td")[0].find_all("strong")[
            0].text.replace('\n', '').replace('\t', '').replace(' ', '')
        h=soup.find_all("div", {"class", "tb_wrap"})[0].find_all("tbody")[i].find_all("span",{"class","td_hour"})[0].text
        if h!='-':
            for j in range(int(n)):
                date.append("2020."+d)

    for i in range(len(times)):
        temp = []
        temp.append(date[i])
        temp.append(times[i])
        when.append(" ".join(temp))

    #############################################################################################
    # 이겼는지 졌는지 파악

    win = 0  # 0=짐, 1=이김, 2=비김
    left_date = []
    right_date = []

    results = soup.find_all("strong", {"class", "td_score"})

    # left
    for index in left_list:

        if results[index].text == "VS":  # 경기취소
            win = 2
        else:
            score = results[index].text.split(':')
            left = int(score[0])
            right = int(score[1])
            if left > right:  # 이김
                win = 1
            elif left < right:  # 짐
                win = 0
            else:  # 비김
                win = 2

        win_dates.append(date[index])
        win_times.append(times[index])
        win_when.append(when[index])
        win_wins.append(win)

    # right
    for index in right_list:
        if results[index].text == "VS":  # 경기취소
            win = 2
        else:
            score = results[index].text.split(':')
            left = int(score[0])
            right = int(score[1])
            if left > right:  # 짐
                win = 0
            elif left < right:  # 이김
                win = 1
            else:  # 비김
                win = 2

        win_dates.append(date[index])
        win_times.append(times[index])
        win_when.append(when[index])
        win_wins.append(win)

d=[]
t=[]
h=[]
m=[]

for date in win_when:
    dt=parse(date)
    d.append(dt.date())
    t_bf = str(dt.time())
    t.append(t_bf)
    h.append(int(t_bf[0:2]))
    m.append(int(t_bf[3:5]))

result = pd.DataFrame()
result["date"] = d
result["time"]=t
result["win"] =win_wins

result.to_csv("ncDions_score.csv", encoding="utf8")

#############################################################################################
# 페이지 이동하기


total_label = []
total_dates = []
total_contents = []
total_titles = []

#############################################################################################
# 페이지 이동하기

for k in range(0,100): #여기가 지금 2페이지까지 해논건데 어디까지 할지 정해야 할듯?
    if k == 0:
        k = ""

    driver.get("https://ncdinos.com/ncboard/fantalk_view?&page=" + str(k) + "0" + "&board_id=freetalk&seq=50043")
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

    id = []
    for href in soup.find("table", class_="bbs_type2").find_all("a"):
        temp = href.attrs['href']
        if temp != "#":
            id.append(temp)

    for i in range(5):
        del id[0]

    #print(id)

    #############################################################################################
    # 게시물 들어가서 날짜, 제목, 본문 크롤링

    for i in id:
        try:
            #print("https://www.heroesbaseball.co.kr/fans/heroesBbs/" + i)
            driver.get("https://ncdinos.com" + i)
        except:
            continue

        time.sleep(0.5)

        tv_source = driver.page_source
        soup = BeautifulSoup(tv_source, "html.parser")

        temp = soup.find("div", {'class': 'subject'}).find_all("strong")[0]
        temp = temp.text
        title = temp.replace("	", "").replace("\n", "")

        #print(title)

        date = soup.find("span", {'class': 'date'})
        date = date.text
        #print(date)

        content = soup.find("div", {'class': 'article'})
        content = content.text
        content = content.replace("	", "")
        content = content.replace("\n", " ")
        #print(content)

        date_d = parse(date).date()
        t_bf = parse(date).time()
        date_t = str(t_bf)
        date_h = int(date_t[0:2])
        date_m = int(date_t[3:5])

        for wd in range(len(d)):
            if win_wins[wd] == 2 or content == " ":
                continue
            if date_d == d[wd] and date_h >= h[wd]:
                if date_h == h[wd] and date_m < m[wd]:
                    continue
                total_titles.append(title)
                total_contents.append(content)
                total_dates.append(date_d)
                total_label.append(win_wins[wd])

driver.quit()

#############################################################################################
# 저장

result = pd.DataFrame()
result["dates"] = total_dates
result["labels"] = total_label
result["titles"] = total_titles
result["reviews"] = total_contents

result.to_csv("ncDinos_Reviews.csv", encoding="utf8", sep="\t")

result = pd.DataFrame()
result["labels"] = total_label
result["titles"] = total_titles

result.to_csv("ncDinos_Reviews_title.csv", encoding="utf8", sep="\t")

result2 = pd.DataFrame()
result2["labels"] = total_label
result2["reviews"] = total_contents

result2.to_csv("ncDinos_Reviews_contents.csv", encoding="utf8", sep="\t")