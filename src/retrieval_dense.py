import os
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
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

import matplotlib.pyplot as plt
import ast

mecab = MeCab.Tagger()
def extract_nouns(text):
    try:
        parsed = mecab.parse(text)
        nouns = []
        for line in parsed.splitlines():
            if '\t' in line:  # MeCab 출력에서 유효한 줄만 처리
                word, feature = line.split('\t')
                if feature.startswith('NNG') or feature.startswith('NNP'):  # 보통명사, 고유명사
                    nouns.append(word)
        return nouns
    except Exception as e:
        print(f"Error during MeCab parsing: {e}")
        return text  # 실패 시 원문 반환


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


def evaluate_metrics_threshold_jaccard(df, retriever, topk=3):
    result_df = df.copy()

    # Retrieval
    result_df['reference'] = ""
    for idx, row in tqdm(result_df.iterrows(), total=len(result_df)):
        # Dense retriever로 상위 topk_base 문서 가져오기
        retrieved_docs = retriever.invoke(row['query'])
        if retrieved_docs:  # 검색된 문서가 있는 경우
            # Jaccard 유사도를 이용해 재정렬
            reranked_docs = jaccard_reranker(row['query'], retrieved_docs, topk=topk)
            references = [ref.page_content for ref in reranked_docs]
            result_df.loc[idx, 'reference'] = str(references)
        else:  # 검색된 문서가 없는 경우
            result_df.loc[idx, 'reference'] = "[]"

    # Metric 계산 초기화
    total_hits = 0  # 전체 hit 수
    total_reciprocal_rank = 0.0  # 전체 reciprocal rank 합계
    total_precision = 0.0  # 전체 precision 합계
    valid_rows = 0  # 유효한 행의 수 (reference가 존재하는 행)

    result_df[['hit', 'rank', 'precision']] = [False, 0, 0.0]

    for idx, row in tqdm(result_df.iterrows(), total=len(result_df)):
        # Reference가 비어 있으면 무시
        if not row['reference'] or row['reference'] == '[]':
            continue

        # 키워드를 쉼표로 분리
        keywords = [kw.strip() for kw in row['keyword'].split(',')]

        # 검색된 문서 리스트
        references = ast.literal_eval(row['reference'])

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


def jaccard_reranker(query, retrieved_docs, topk=5):
    # Jaccard 유사도 기반 재정렬
    tokenized_query = extract_nouns(query)
    context_scores = []
    for doc in retrieved_docs:
        tokenized_context = extract_nouns(doc.page_content)
        common_terms = set(tokenized_query).intersection(set(tokenized_context))
        score = len(common_terms) / len(set(tokenized_query).union(set(tokenized_context))) 
        context_scores.append(score)

    # 유사도 점수에 따라 컨텍스트 재정렬 후 상위 topk 반환
    reranked_docs = [doc for _, doc in sorted(zip(context_scores, retrieved_docs), key=lambda x: x[0], reverse=True)][:topk]
    
    return reranked_docs


def process_row(row):
    problems = literal_eval(row['problems'])
    paragraph = row['paragraph']
    question = problems['question']
    choices = problems['choices']
    choices_str = " ".join(choices)
    
    query = prompt.format(paragraph=paragraph, question=question, choices=choices_str)
    
    # 보수적 기준: 길이가 500자 이상인 문서 제외
    if len(str(paragraph)) > 500:
        return []
    
    # Dense retriever로 문서 검색
    docs = faiss_retriever.invoke(query)
    
    if docs:
        # Jaccard 유사도로 재정렬
        reranked_docs = jaccard_reranker(query, docs, topk=topk)
        # 최종 상위 topk 문서 추출
        retrieved_docs = [doc.page_content for doc in reranked_docs]
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

# RAG Data Loading 
rag_folder = "/data/ephemeral/home/workspace/contest_baseline_code/data/rag"
rag_files = glob(f"{rag_folder}/*.csv")

# Concatenate RAG data
rag_data_source = [pd.read_csv(file) for file in rag_files]
rag_data = pd.concat(rag_data_source, axis=0, ignore_index=True)
print(f"RAG Data Count: {rag_data.shape[0]}")

# Use only documents with at least 25 characters
rag_data = rag_data[rag_data.context.str.len() >= 25]
print(f"filtered (len > 25) RAG Data Count: {rag_data.shape[0]}")

# Rag Data Chunking
loader = DataFrameLoader(rag_data, page_content_column='context')
documents = loader.load()

# Chunking
chunk_size = 800
chunk_overlap = 200

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator=". ",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    encoding_name='cl100k_base'
)
split_docs = text_splitter.split_documents(tqdm(documents))
print(f"Chunked Document Count: {len(split_docs)}")

# load vector DB
model_name = 'dragonkue/BGE-m3-ko'
device = 'cuda'

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True},
)

vector_store_path = f"/data/ephemeral/home/workspace/contest_baseline_code/data/db/faiss_{model_name.replace('/', '_')}_chunk-{chunk_size}-{chunk_overlap}_v2"

# Load existing vector store or create new one
if os.path.exists(vector_store_path):
    print("Loading existing vector store...")
    vector_store = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
        )
else:
    print("Creating new vector store...")
    vector_store = FAISS.from_documents(
        [split_docs[0]],
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )

    batch_size = 4  # 적절한 배치 크기로 조절 가능
    docs_to_add = split_docs[1:]
    with tqdm(total=len(docs_to_add), desc="Ingesting documents") as pbar:
        for i in range(0, len(docs_to_add), batch_size):
            batch_docs = docs_to_add[i:i + batch_size]
            vector_store.add_documents(batch_docs)
            pbar.update(len(batch_docs))
            torch.cuda.empty_cache()
            
    vector_store.save_local(vector_store_path)

doc_count = vector_store.index.ntotal
print(f"Document Count in Vector Store: {doc_count}")

topk_base = 15
topk = 3
score_threshold = 0.4

faiss_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":topk_base, "score_threshold": score_threshold},
)

# faiss_retriever = vector_store.as_retriever(search_kwargs={"k":topk})

# result_df, hit, mrr, avg_precision = evaluate_metrics_threshold_jaccard(eval_set, faiss_retriever, topk)
# valid_df = result_df[result_df.reference != "[]"]
# # print(f"Score Threshold: {score_threshold}")
# print(f"Hit@{topk}: {hit:.4f}")
# print(f"MRR@{topk}: {mrr:.4f}")
# print(f"Precision@{topk}: {avg_precision:.4f}")
# print(f"문서 개수: {len(valid_df)}")

# test set 저장
tqdm.pandas()

target_data = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/preprocessed/train_fix_khan_kor_v2_korean.csv")

prompt = "{paragraph} {question} {choices}"
    
# query 생성
target_data['reference'] = target_data.progress_apply(process_row, axis=1)

# 결과 저장
target_data.to_csv(f"/data/ephemeral/home/workspace/contest_baseline_code/data/preprocessed/train_rag_rerank{topk}_final.csv", index=False)

# test set 저장
target_data = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/raw/test.csv")
prompt = "{paragraph} {question} {choices}"

# query 생성
target_data['reference'] = target_data.progress_apply(process_row, axis=1)

# 결과 저장
target_data.to_csv(f"/data/ephemeral/home/contest_baseline_code/data/preprocessed/test_rag_rerank{topk}_v2_list.csv", index=False)