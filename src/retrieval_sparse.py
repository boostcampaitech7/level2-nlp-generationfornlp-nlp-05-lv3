import os
import numpy as np
import pandas as pd
import MeCab
import faiss
import torch
from tqdm import tqdm
from glob import glob
from ast import literal_eval
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import BM25Retriever
# from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from typing import List

import matplotlib.pyplot as plt


# Define the Document class
class Document:
    def __init__(self, page_content, score):
        self.page_content = page_content
        self.score = score

# Modify CustomBM25Retriever
class CustomBM25Retriever:
    def __init__(self, bm25_instance, documents_df, topk=5, score_threshold=0.4):
        """
        Custom BM25 Retriever that fetches top-k documents above a score threshold.

        Args:
            bm25_instance (BM25Okapi): An instance of BM25Okapi.
            documents_df (pd.DataFrame): DataFrame containing the documents.
            topk (int): Number of top documents to retrieve.
            score_threshold (float): Minimum BM25 score to consider.
        """
        self.bm25 = bm25_instance
        self.documents_df = documents_df
        self.topk = topk
        self.score_threshold = score_threshold

    def retrieve(self, query) -> List[Document]:
        """
        Retrieves documents relevant to the query.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: List of Document instances containing document content and score.
        """
        processed_query = extract_nouns(query)
        if not processed_query:
            print("검색어에서 명사를 추출할 수 없습니다.")
            return []

        scores = self.bm25.get_scores(processed_query)
        top_k_indices = np.argsort(scores)[::-1][:self.topk]
        results = []
        for idx in top_k_indices:
            score = scores[idx]
            if score >= self.score_threshold:
                original_document = self.documents_df.iloc[idx]['context']
                # Create a Document instance
                doc = Document(
                    page_content=original_document,
                    score=min(score / 100, 1)
                )
                results.append(doc)
        return results

    def invoke(self, query):
        """
        Alias for the retrieve method to maintain compatibility.

        Args:
            query (str): The search query.

        Returns:
            List[Document]: Retrieved documents with scores.
        """
        return self.retrieve(query)

# Initialize MeCab Tagger
mecab = MeCab.Tagger()
def extract_nouns(text):
    """
    Extracts nouns from the given text using MeCab.
    
    Args:
        text (str): The text to process.
        
    Returns:
        List[str]: A list of extracted nouns.
    """
    try:
        parsed = mecab.parse(text)
        nouns = []
        for line in parsed.splitlines():
            if '\t' in line:
                word, feature = line.split('\t')
                if feature.startswith('NN'):  # Nouns in Korean
                    nouns.append(word)
        return nouns
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return []

def evaluate_metrics_threshold(df, retriever):
    result_df = df.copy()

    # Retrieval
    result_df['reference'] = ""
    for idx, row in tqdm(result_df.iterrows()):
        retrieved_docs = retriever.invoke(row['query'])
        if retrieved_docs:  # 검색된 문서가 있는 경우
            references = [ref.page_content for ref in retrieved_docs]
            result_df.loc[idx, 'reference'] = str(references)
        else:  # 검색된 문서가 없는 경우
            result_df.loc[idx, 'reference'] = ""

    # Metric 계산 초기화
    total_hits = 0  # 전체 hit 수
    total_reciprocal_rank = 0.0  # 전체 reciprocal rank 합계
    total_precision = 0.0  # 전체 precision 합계
    valid_rows = 0  # 유효한 행의 수 (reference가 존재하는 행)

    result_df[['hit', 'rank', 'precision']] = [False, 0, 0.0]

    for idx, row in tqdm(result_df.iterrows()):
        # Reference가 비어 있으면 무시
        if not row['reference'] or row['reference'] == '[]':
            continue

        # 키워드를 쉼표로 분리
        keywords = [kw.strip() for kw in row['keyword'].split(',')]

        # 검색된 문서 리스트
        references = eval(row['reference'])

        K = len(references)  # 검색된 문서 수 (Top K)
        relevant_retrieved_docs = 0  # 검색 결과 중 관련 문서의 수 초기화
        rank = 0  # 관련 문서의 첫번째 등장 순위 초기화
        found = False

        for i, doc in enumerate(references):
            # 문서의 공백 제거
            doc_no_space = doc.replace(' ', '').replace('\n', '')
            doc_is_relevant = False
            for kw in keywords:
                # 키워드의 공백 제거
                kw_no_space = kw.replace(' ', '')
                # 키워드가 문서에 포함되어 있는지 확인
                if kw_no_space in doc_no_space:
                    doc_is_relevant = True
                    if not found:
                        rank = i + 1  # 순위는 1부터 시작
                        found = True
                    break  # 키워드를 찾았으므로 내부 루프 종료
            if doc_is_relevant:
                relevant_retrieved_docs += 1  # 관련된 문서의 수 1 증가

        if rank > 0:
            result_df.loc[idx, 'hit'] = True
            result_df.loc[idx, 'rank'] = rank
            total_hits += 1  # hit 증가
            reciprocal_rank = 1.0 / rank  # 역순위 계산
            total_reciprocal_rank += reciprocal_rank  # 역순위 합계에 추가

        # 현재 쿼리에 대한 precision 계산
        precision = relevant_retrieved_docs / K if K > 0 else 0
        result_df.loc[idx, 'precision'] = precision
        total_precision += precision

        valid_rows += 1  # reference가 존재하는 유효한 행의 수 증가

    # 전체 metric 계산
    if valid_rows > 0:
        hit_at_k = total_hits / valid_rows
        mrr_at_k = total_reciprocal_rank / valid_rows
        avg_precision = total_precision / valid_rows
    else:
        hit_at_k = 0.0
        mrr_at_k = 0.0
        avg_precision = 0.0

    return result_df, hit_at_k, mrr_at_k, avg_precision

def process_row(row):
    problems = literal_eval(row['problems'])
    paragraph = row['paragraph']
    question = problems['question']
    choices = problems['choices']
    choices_str = " ".join(choices)
    
    query = prompt.format(paragraph=paragraph, question=question, choices=choices_str)
    
    if len(str(paragraph)) > 500: # 보수적 기준: 약 600~700자, 널널한 기준: 약 300~400자
        return []
    # sparse 사용
    docs = bm25_retriever.retrieve(query)
    if docs:
        retrieved_docs = [doc.page_content for doc in docs]
        return retrieved_docs
    else:
        return []

# Retrieval evaluation dataset
eval_set = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/rag_eval/external_knowledge_w_label_keyword.csv")
prompt = "{paragraph}\n{question}\n{choices}"
eval_set['query'] = ""  # input 형태 맞추기
for idx, row in eval_set.iterrows():
    problems = literal_eval(row['problems'])
    question = problems['question']
    choices = problems['choices']
    choices_str = " ".join(choices)
    query = prompt.format(paragraph=row['paragraph'], question=question, choices=choices_str)
    eval_set.loc[idx, 'query'] = query


rag_folder = "/data/ephemeral/home/workspace/contest_baseline_code/data/rag"
rag_files = glob(f"{rag_folder}/*.csv")

# Concatenate RAG data
rag_data_source = [pd.read_csv(file) for file in rag_files]
rag_data = pd.concat(rag_data_source, axis=0, ignore_index=True)
print(f"RAG Data Count: {rag_data.shape[0]}")

# Use only documents with at least 25 characters
rag_data = rag_data[rag_data.context.str.len() >= 25]
print(f"filtered (len > 25) RAG Data Count: {rag_data.shape[0]}")

loader = DataFrameLoader(rag_data, page_content_column='context')
documents = loader.load()

# Chunking
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator=". ",
    chunk_size=800,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)
split_docs = text_splitter.split_documents(tqdm(documents))
print(f"Chunked Document Count: {len(split_docs)}")

# Convert split_docs back to DataFrame
rag_data_chunk = pd.DataFrame(
    [{
        **doc.metadata,              # Include all metadata from the original document
        'context': doc.page_content,  # The chunked content
    } for doc in split_docs]
)

# Extract nouns from each document's context
rag_data_chunk['nouns'] = rag_data_chunk['context'].apply(extract_nouns)

# Handle cases where noun extraction failed
rag_data_chunk['nouns'] = rag_data_chunk['nouns'].apply(lambda x: x if x else [])

bm25 = BM25Okapi(rag_data_chunk['nouns'].tolist())

# Configuration parameters
topk = 3
score_threshold = 0.4

# Initialize the custom retriever
bm25_retriever = CustomBM25Retriever(
    bm25_instance=bm25,
    documents_df=rag_data_chunk,
    topk=topk,
    score_threshold=score_threshold
)

# Evaluate the BM25 retriever
query = """
선비들 수만 명이 대궐 앞에 모여 만 동묘와 서원을 다시 설립할 것을 청하니, (가)이/가 크게 노하여 한성부의 조례(皂隷)와 병졸로 하여 금 한 강 밖으로 몰아내게 하고 드디어 천여 곳의 서원을 철폐하고 그 토지를 몰수하여 관에 속하게 하였다.－대한계년사 －

(가) 인물이 추진한 정책으로 옳지 않은 것은?

1 - 사창제를 실시하였다 .
2 - 대전회통을 편찬하였다 .
3 - 비변사의 기능을 강화하였다 .
4 - 통상 수교 거부 정책을 추진하였다 .
"""

# Retrieve relevant documents
search_results = bm25_retriever.retrieve(query)

# Display the results
print('-' * 25, "BM25 Retrieval Results", '-' * 25)
print("문제:")
print(query)

print("검색 결과:")
for i, result in enumerate(search_results, 1):
    print(f"Result {i}: (Score: {result.score:.4f}) {result.page_content}")

# # Evaluate the BM25 retriever
# result_df, hit, mrr, avg_precision = evaluate_metrics_threshold(eval_set, bm25_retriever)
# valid_df = result_df[result_df.reference != ""]
# # print(f"Score Threshold: {score_threshold}")
# print(f"Hit@{topk}: {hit:.4f}")
# print(f"MRR@{topk}: {mrr:.4f}")
# print(f"Precision@{topk}: {avg_precision:.4f}")
# print(f"문서 개수: {len(valid_df)}")


# train set 저장
tqdm.pandas()

target_data = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/preprocessed/train_fix_khan_kor_v2_korean.csv")
prompt = "{paragraph} {question} {choices}"
    
# query 생성
target_data['reference'] = target_data.progress_apply(process_row, axis=1)

# 결과 저장
target_data.to_csv(f"/data/ephemeral/home/contest_baseline_code/data/preprocessed/train_rag_sparse{topk}_v2_list.csv", index=False)


# test set 저장
target_data = pd.read_csv("/data/ephemeral/home/contest_baseline_code/data/raw/test.csv")
prompt = "{paragraph} {question} {choices}"

# query 생성
target_data['reference'] = target_data.progress_apply(process_row, axis=1)

# 결과 저장
target_data.to_csv(f"/data/ephemeral/home/contest_baseline_code/data/preprocessed/test_rag_sparse{topk}_v2_list.csv", index=False)