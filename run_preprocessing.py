from src.wikipedia_preprocessing import WikipediaPreprocessing

preprocessor = WikipediaPreprocessing()

# 위키 파일 파싱 후 CSV로 저장
filepaths = preprocessor.get_filepaths('text')
df_parsed = preprocessor.parse_all_files(filepaths, 'wiki_parsed.csv')

# 파싱된 데이터 전처리 및 CSV 저장
df_preprocessed = preprocessor.preprocess('wiki_parsed.csv', 'wiki_cleaned.csv')
