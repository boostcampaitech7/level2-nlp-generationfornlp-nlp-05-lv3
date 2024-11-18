import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
logging.basicConfig(level="INFO")


class OpenStaxCrawling:
    def __init__(self, subject):
        self.url_base = 'https://openstax.org/books/'
        self.pages = []
        self._set_subject(subject)
    
    # 과목 설정 및 페이지 목록 초기화
    def _set_subject(self, subject):
        subjects = {
            "psychology": {
                'url_subject': 'psychology-2e/pages/',
                'url_initial': 'https://openstax.org/books/psychology-2e/pages/1-introduction',
                'summary_page': "summary",
                "key_term_page": "key-terms",
                "chapter_len": 16
            },
            "economics": {
                'url_subject': 'principles-economics-3e/pages/',
                'url_initial': 'https://openstax.org/books/principles-economics-3e/pages/1-introduction',
                'summary_page': "summary",
                "key_term_page": "key-concepts-and-summary",
                "chapter_len": 34
            },
            "us_history": {
                'url_subject': 'us-history/pages/',
                'url_initial': 'https://openstax.org/books/us-history/pages/1-introduction',
                'summary_page': "summary",
                "key_term_page": "key-terms",
                "chapter_len": 32
            },
            "world_history1":{
                'url_subject': 'world-history-volume-1/pages/',
                'url_initial': 'https://openstax.org/books/world-history-volume-1/pages/1-introduction',
                'summary_page': "section-summary",
                "key_term_page": "key-terms",
                "chapter_len": 17
            },
            "world_history2":{
                'url_subject': 'world-history-volume-2/pages/',
                'url_initial': 'https://openstax.org/books/world-history-volume-2/pages/1-introduction',
                'summary_page': "section-summary",
                "key_term_page": "key-terms",
                "chapter_len": 15
            },
            "politics":{
                'url_subject': 'introduction-political-science/pages/',
                'url_initial': 'https://openstax.org/books/introduction-political-science/pages/1-introduction',
                'summary_page': "summary",
                "key_term_page": "key-terms",
                "chapter_len": 16
            }
        }
        if subject in subjects:
            self.url_base += subjects[subject]['url_subject']
            self.pages = self._get_filtered_openstax_links(subjects[subject]['url_initial'])
            self.summary_page = subjects[subject]['summary_page']
            self.key_term_page = subjects[subject]['key_term_page']
            self.chapter_len = subjects[subject]['chapter_len']
        else:
            raise ValueError("Unsupported subject!")
    
    def _get_filtered_openstax_links(self, url):
        # 웹 페이지 요청
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 'li' 태그에서 'data-type="page"' 속성을 가진 요소 찾기
        chapters = soup.find_all('li', {'data-type': 'page'})
        
        # 링크와 텍스트 추출
        links = []
        for chapter in chapters:
            a_tag = chapter.find('a', href=True)
            if a_tag:
                href = a_tag['href']
                # 'https://openstax.org' 뒤의 텍스트만 추출
                text_part = href.replace('https://openstax.org', '')
                links.append(text_part)
        
        # '숫자-숫자-텍스트' 형식만 필터링
        filtered_links = [link for link in links if re.match(r'^\d+-\d+-[a-zA-Z\-]+', link)]
        
        return filtered_links
    
    # 페이지 이름에서 숫자와 하이픈 제거하고 소문자로 변환
    def extract_text(self, input_string):
        cleaned_text = re.sub(r'^\d+-\d+-', '', input_string).replace('-', ' ')
        return cleaned_text.strip().lower()

    # 페이지 요청
    def fetch_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            logging.warning(f"Error fetching {url}: {e}")
            return None
    
    # 본문 크롤링
    def crawl_content(self, soup, page_name):
        if not soup:
            return pd.DataFrame()

        titles, contexts = [], []
        current_h2, current_h3 = None, None

        for section in soup.find_all('section', {'data-depth': '1'}):
            # 제목을 찾고 본문을 연결
            for element in section.find_all(['h2', 'h3', 'p']):
                text = element.get_text(strip=True)
                if element.name == 'h2':
                    current_h2, current_h3 = text, None
                elif element.name == 'h3':
                    current_h3 = text
                elif element.name == 'p':
                    titles.append(current_h3 or current_h2)
                    contexts.append(text)

        return pd.DataFrame({'section': page_name, 'title': titles, 'context': contexts})

    # 용어 및 정의 크롤링
    def crawl_key_terms(self, soup):
        if not soup:
            return pd.DataFrame()

        terms, definitions = [], []
        for dl in soup.find_all('dl'):
            for term, definition in zip(dl.find_all('dt'), dl.find_all('dd')):
                terms.append(term.get_text(strip=True))
                definitions.append(definition.get_text(strip=True))
        
        return pd.DataFrame({'section': 'term', 'title': terms, 'context': definitions})

    # 페이지 목록 순회하며 크롤링
    def crawl_pages(self, url_func, page_list, section_name=''):
        total_df = pd.DataFrame()
        for idx, page in enumerate(page_list):
            url = url_func(page)
            soup = self.fetch_page(url)
            if section_name == 'term':
                df = self.crawl_key_terms(soup)
            else:
                page_name = self.extract_text(page)
                df = self.crawl_content(soup, page_name)
            total_df = pd.concat([total_df, df], ignore_index=True)
            logging.info(f"Crawled {section_name or page} (index: {idx+1}/{len(page_list)})")
            time.sleep(2)
        return total_df
    
    # 모든 데이터 크롤링하여 csv로 저장
    def crawl(self, save_path):
        # main contents
        main_df = self.crawl_pages(lambda page: f"{self.url_base}{page}", self.pages)
        
        # summary contents
        summary_pages = [f"{i+1}-{self.summary_page}" for i in range(self.chapter_len)]
        summary_df = self.crawl_pages(lambda page: f"{self.url_base}{page}", summary_pages, 'summary')
        
        # key terms
        key_term_pages = [f"{i+1}-{self.key_term_page}" for i in range(self.chapter_len)]
        key_terms_df = self.crawl_pages(lambda page: f"{self.url_base}{page}", key_term_pages, 'term')

        # 합치기 및 저장
        total_df = pd.concat([main_df, summary_df, key_terms_df], ignore_index=True)
        total_df.to_csv(save_path, index=False)
        logging.info(f"Data saved to {save_path}")
        
        
# 사용 예시
if __name__ == "__main__":
    subject = "psychology"  # psychology, economics, us_history, world_history1, world_history2, politics
    crawler = OpenStaxCrawling(subject)
    crawler.crawl(f"openstax_{subject}.csv")