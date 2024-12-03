import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging 
from tqdm import tqdm
logging.basicConfig(level="INFO")


class KoreanHistoryBookCrawling:
    def __init__(self, option):
        self.url_base = 'http://contents.history.go.kr/front/'
        self.links = []
        self._set_option(option)
        self._get_links(option)
    
    # 과목 설정 및 페이지 목록 초기화
    def _set_option(self, option):
        options = {
            "textbook": {
                'url_option': 'ta/view.do?levelId=ta_h71_',
            },
            "term": {
                'url_option': '',
            },
        }
        if option in options:
            self.url_base += options[option]['url_option']
        else:
            raise ValueError("Unsupported subject!")
    
    def _get_links(self, option):
        valid_links = []
        if option == "textbook":
            links = [
                f"{self.url_base}{d1:04d}_{d2:04d}_{d3:04d}_{d4:04d}"
                for d1 in range(30, 71, 10)
                for d2 in range(10, 51, 10)
                for d3 in range(10, 61, 10)
                for d4 in range(10, 71, 10)
            ]
            for link in tqdm(links):
                response = requests.get(link, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Check if h1 tag has content
                h1_tag = soup.find('h1')
                if h1_tag and h1_tag.get_text(strip=True):  # h1 exists and has text
                    valid_links.append(link)
                    
            self.links = valid_links

    # 페이지 요청
    def fetch_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logging.warning(f"Error fetching {url}: {e}")
            return None
    
    # 단원명 추출
    def extract_section_name(self, url):
        soup = self.fetch_page(url)
        matching_url = url.split("_")[-4:-1]
        matching_url = "_".join(matching_url)
        # 'lnb' 클래스 내에서 텍스트와 링크를 찾기
        section = soup.find('section', class_='lnb')
        for link in section.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # 단원명 추출 및 번호 삭제
            if href.endswith(matching_url):
                text_without_number = re.sub(r'\[\d+\] ', '', text)
                return text_without_number
    
    # 본문 크롤링
    def crawl_content(self, soup, section_name):
        if not soup:
            return pd.DataFrame()

        # 제목(title) 추출
        title = soup.find('h1').get_text(strip=True)
        
        # 본문(context) 추출
        context_list = []
        
        # 일반 <p> 태그 내용 추출
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:  # 빈 텍스트 필터링
                context_list.append(text)
                    
        # 예외 상자 내 제목과 본문 연결 추출
        for box in soup.select('.annotation_tbk, .notab_annotation'):
            sub_context = box.get_text(strip=True, separator=' ')
            context_list.append(f"{sub_context}")
            
        return pd.DataFrame({"section": section_name, "title": title, "context": context_list})

    # 페이지 목록 순회하며 크롤링
    def crawl(self, save_path, section_name=""):
        total_df = pd.DataFrame()
        for idx, url in enumerate(tqdm(self.links)):
            soup = self.fetch_page(url)
            if section_name == "":
                section_name = self.extract_section_name(url)
            df = self.crawl_content(soup, section_name)
            total_df = pd.concat([total_df, df], ignore_index=True)
            time.sleep(2)
        
        total_df.to_csv(save_path, index=False)
        logging.info(f"Data saved to {save_path}")
        
        
# 사용 예시
if __name__ == "__main__":
    option = "textbook"  # textbook, term
    crawler = KoreanHistoryBookCrawling(option)
    crawler.crawl(f"korean_history_{option}.csv")
