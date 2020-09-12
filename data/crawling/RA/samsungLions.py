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

total_dates = []
total_contents = []
total_titles = []
date_temp=[]

#############################################################################################
# 페이지 이동하기

for k in range(1,21): #시험삼아 20페이지만 했음
    driver.get("https://www.samsunglions.com/fan/fan01.asp?page=" + str(k) + "&keyword=&search=&myarticle=&oa=1&ob=1")
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

    ind = soup.find_all("td", {"class": "tit"})
    id = []
    id = [tv.select("a[href]")[0]["href"] for i, tv in enumerate(ind)]
    #print(id)

    #############################################################################################
    #게시물 들어가서 날짜, 시간, 제목, 본문 크롤링

    count = 0
    for i in id:
        count = count + 1
        if k==1:
            if count < 6:  # 1페이지의 공지 게시물 부분 건너뛰기
                continue
        try:
            print("https://www.samsunglions.com/" + i)
            driver.get("https://www.samsunglions.com/" + i)
        except:
            continue

        time.sleep(0.5)

        tv_source = driver.page_source
        soup = BeautifulSoup(tv_source, "html.parser")

        date = soup.find_all("em",{"class","c"}) #날짜, 시간
        #print(date)
        title = soup.find("div", {'class': 'tit'})  # 제목
        # print(title)
        content = soup.find("div", {'class': 'con'}).find_all("p")  # 본문
        #print(content)

        one_date_temp=date[2].text
        one_date=date[2].text.replace("오전",' ').replace("오후",' ')
        #print(one_date)

        one_title = title.select("h4")[0].text.replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').replace("새글",'')
        #print(one_title)

        try:
            one_content = ""
            for p in content:
                one_content = one_content + p.text.replace('\n', '').replace('\t', '').replace('  ', '').replace('\"','') + " "
            #print(one_content)
        except:
            continue

        total_titles.append(one_title)
        total_contents.append(one_content)
        total_dates.append(one_date)
        date_temp.append(one_date_temp)

driver.quit()

d=[]
t=[]
dtdt=[]

count=0
for date in total_dates:
    dt=parse(date)
    d.append(dt.date())
    t.append(dt.time())

    if "오후" in date_temp[count]:
        print(date_temp[count])
        a=datetime.datetime.combine(datetime.date(1,1,1),t[count])
        b=a+datetime.timedelta(hours=12)
        t[count]=b.time()
        if t[count].hour==00:
            t[count]=a.time()

    elif "오전" in date_temp[count]:
        print(date_temp[count])
        a = datetime.datetime.combine(datetime.date(1, 1, 1), t[count])
        t[count] = a.time()
        if t[count].hour == 12:
            b = a + datetime.timedelta(hours=12)
            t[count] = b.time()

    count=count+1

#############################################################################################
# 저장


result = pd.DataFrame()
result["win"]=d
result["date"] = d
result["time"]=t
result["titles"]=total_titles
result["reviews"] = total_contents

result.to_csv("samsungLions_Reviews.txt", encoding="utf8", sep="\t")
result.to_csv("samsungLions_Reviews.csv", encoding="utf8")

df1 = pd.read_csv('samsung_score.csv')
D1=df1['date'].tolist()
T1=df1['time'].tolist()
W=df1['win']

df = pd.read_csv('samsungLions_Reviews.csv')
D2=df['date'].tolist()
T2=df['time'].tolist()

DT1=df['date']

print(D1)
print(D2)

df['win']=D2

print(df)


for i in range(len(D2)):
    temp=D1.index(D2[i])
    print(temp)
    print("win?: ",W[temp])
    if T1[temp] < T2[i]: #일단 경기 당일 경기 시작 이후부터 자정까지의 게시글로 했음
        df['win'][i] = W[temp]
    else: #그외 게시글
        df['win'][i]=9999

for i in range(len(D2)): #비기거나 취소된 경기, 그리고 시간조건에 맞지 않는 그외 게시글들 드랍
    if df['win'][i]>1:
        df=df.drop(i,axis=0)

df=df.drop("Unnamed: 0",axis=1)
print(df)

df.to_csv("samsungLions_Reviews.txt", encoding="utf8",sep='\t')

df_title= pd.DataFrame()
df_contents= pd.DataFrame()

df_title=df.drop(["reviews","date","time"],axis=1)
df_title.rename(columns = {'' : 'id','win':'label','title':'review'}, inplace = True)
print(df_title)
df_title.reindex(columns=['id', 'review', 'label'])
df_title= df_title.reset_index(drop=True)

df_contents=df.drop(["titles","date","time"],axis=1)
df_contents.rename(columns = {'' : 'id','win':'label','title':'review'}, inplace = True)
print(df_title)
df_contents.reindex(columns=['id', 'review', 'label'])
df_contents = df_contents.reset_index(drop=True)

df_title.to_csv("samsung_reviews_title.txt", encoding="utf8",sep='\t')
df_contents.to_csv("samsung_reviews_contents.txt", encoding="utf8",sep='\t')
