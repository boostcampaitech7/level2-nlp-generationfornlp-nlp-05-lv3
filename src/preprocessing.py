import os
import re
import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
logging.basicConfig(level="INFO")

# wikipedia 덤프 파일 전처리
class WikipediaPreprocessing:
    def __init__(self):
        pass

    # 폴더 내 모든 파일 경로를 리스트로 반환
    def get_filepaths(self, dirname):
        filepaths = []
        for root, _, files in os.walk(dirname):
            for filename in files:
                if re.match(r"wiki_[0-9][0-9]", filename):
                    filepaths.append(os.path.join(root, filename))
        return sorted(filepaths)

    # 단일 파일에서 doc_id, url, title, context 추출
    def parse_single_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()

        pattern = r'<doc id="(\d+)" url="([^"]+)" title="([^"]+)">(.*?)</doc>'
        matches = re.findall(pattern, content, re.DOTALL)
        data = [{"doc_id": doc_id, "url": url, "title": title, "context": context.strip()} for doc_id, url, title, context in matches]
        return pd.DataFrame(data)

    # 여러 파일을 파싱하여 단일 DataFrame 생성 및 CSV 저장
    def parse_all_files(self, filepaths, output_path):
        all_data = []
        for filepath in tqdm(filepaths, desc="Parsing wikipedia documents"):
            df = self.parse_single_file(filepath)
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"저장 완료: {output_path}")
        return combined_df

    # 전처리: context에서 title 제거
    def remove_title_prefix(self, df):
        def _remove_prefix(row):
            prefix = f"{row['title']}\n\n"
            return row['context'][len(prefix):] if row['context'].startswith(prefix) else row['context']

        df['context'] = df.apply(_remove_prefix, axis=1)
        return df

    # 전처리: context 정제 (특수 패턴 제거)
    def clean_text(self, text):
        # 개행문자 처리: \n, \\n 빈칸으로 대치
        text = re.sub(r'\\n|\n', ' ', text)

        # [[분류:...]] 패턴 제거
        text = re.sub(r'\[\[분류:.*?\]\]', ' ', text)

        # [[원본 문서 링크|별명]]에서 [[별명]]만 남기기
        while re.search(r'\[\[.*?\|.*?\]\]', text):
            text = re.sub(r'\[\[(?:[^\[\]]*\|)*(.*?)\]\]', r'\1', text)

        # 대괄호 제거: [[내용]] -> 내용
        text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)

        # 중복 띄어쓰기 하나의 공백으로 대치
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    # DataFrame 내 context 전처리
    def preprocess_context(self, df):
        df['context'] = df['context'].apply(self.clean_text)
        return df

    # context가 100글자 미만인 문서 제거
    def filter_short_contexts(self, df, min_length=50):
        initial_count = len(df)
        df['context_len'] = df['context'].str.len()
        df = df[df['context_len'] >= min_length]
        logging.info(f"{min_length}글자 미만 문서 제거: {initial_count - len(df)}건")
        return df.drop(columns='context_len')

    # 중복 context 제거
    def remove_duplicates(self, df):
        initial_count = len(df)
        df = df.drop_duplicates(subset=['context'], keep='first')
        logging.info(f"중복 제거: {initial_count - len(df)}건")
        return df

    # 전체 전처리 파이프라인
    def preprocess(self, data_path, output_path):
        logging.info("데이터 로드 중...")
        df = pd.read_csv(data_path)
        initial_len = len(df)

        logging.info("NA 값 제거 중...")
        df = df.dropna(subset=['title', 'context'])

        logging.info("제목 제거 중...")
        df = self.remove_title_prefix(df)

        logging.info("텍스트 전처리 중...")
        df = self.preprocess_context(df)

        logging.info("짧은 문서 제거 중...")
        df = self.filter_short_contexts(df)

        logging.info("중복 문서 제거 중...")
        df = self.remove_duplicates(df)

        logging.info(f"최초 데이터 수: {initial_len}")
        logging.info(f"최종 데이터 수: {len(df)}")
        logging.info(f"최종 데이터 저장: {output_path}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return df    
    
    
# OpenStax 교과서 데이터 크롤링
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


# 우리역사넷 국사 교과서 크롤링
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
    ########################################
    #     wikipedia 데이터 파싱 및 전처리       #
    ########################################
    preprocessor = WikipediaPreprocessing()
    
    # 위키 파일 파싱 후 CSV로 저장
    filepaths = preprocessor.get_filepaths('text')
    df_parsed = preprocessor.parse_all_files(filepaths, 'wiki_parsed.csv')

    # 파싱된 데이터 전처리 및 CSV 저장
    df_preprocessed = preprocessor.preprocess('wiki_parsed.csv', 'wiki_cleaned.csv')
    
    
    ########################################
    #       OpenStax 교과서 데이터 크롤링       #
    ########################################
    subject = "psychology"  # psychology, economics, us_history, world_history1, world_history2, politics
    crawler = OpenStaxCrawling(subject)
    crawler.crawl(f"openstax_{subject}.csv")
    
    
    ########################################
    #       우리역사넷 교과서 데이터 크롤링       #
    ########################################
    option = "textbook"  # textbook, term
    crawler = KoreanHistoryBookCrawling(option)
    crawler.crawl(f"korean_history_{option}.csv")
    
    # 교과서 용어 사전 데이터 추가 크롤링
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
    
    