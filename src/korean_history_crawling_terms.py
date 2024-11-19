import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time


# 용어 사전 링크 추출하기
url = "http://contents.history.go.kr/front/tg/list.do?treeId=0100"
response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')

# 링크 추출 (ul 태그 내 li > a 태그)
links = soup.select('ul.list_type1.mt15 li a')

# 결과 저장
collected_links = []
for link in links:
    href = link.get('href')
    if href:
        full_url = requests.compat.urljoin(url, href)  # 상대경로를 절대경로로 변환
        collected_links.append(full_url)
        
def fetch_page(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')



result_df = pd.DataFrame()
for idx, link in enumerate(tqdm(collected_links)):
    soup = fetch_page(link)
    
    # 제목 추출
    title = soup.find('h1').get_text(strip=True)
    
    # 본문 추출 (정의 및 내용 본문)
    context_list = []
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        if text:  # 빈 텍스트 필터링
            context_list.append(text)
    context_list.pop()
    
    df = pd.DataFrame({"section": "term", "title": title, "context": context_list})
    result_df = pd.concat([result_df, df], axis=0, ignore_index=True)
    time.sleep(2)
    if idx % 10 == 0:
        result_df.to_csv("korean_history_term.csv", index=False)
result_df.to_csv("korean_history_term.csv", index=False)