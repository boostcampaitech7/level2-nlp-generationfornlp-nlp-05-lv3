import os
import re
from tqdm import tqdm
import pandas as pd
import logging

logging.basicConfig(level="INFO")

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

from src.wikipedia_preprocessing import WikipediaPreprocessing

preprocessor = WikipediaPreprocessing()

# 위키 파일 파싱 후 CSV로 저장
filepaths = preprocessor.get_filepaths('text')
df_parsed = preprocessor.parse_all_files(filepaths, 'wiki_parsed.csv')

# 파싱된 데이터 전처리 및 CSV 저장
df_preprocessed = preprocessor.preprocess('wiki_parsed.csv', 'wiki_cleaned.csv')