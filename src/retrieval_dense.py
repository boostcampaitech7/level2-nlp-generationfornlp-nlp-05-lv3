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
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
tqdm.pandas()

# Initialize MeCab for Korean tokenization
mecab = MeCab.Tagger()
def extract_nouns(text):
    """
    Extract nouns from the given text using MeCab.

    Args:
        text (str): Input text.
    
    Returns:
        list: A list of extracted nouns.
    """
    try:
        parsed = mecab.parse(text)
        nouns = []
        for line in parsed.splitlines():
            if '\t' in line:  # Process valid lines in MeCab output
                word, feature = line.split('\t')
                if feature.startswith('NNG') or feature.startswith('NNP'):  # Common or proper nouns
                    nouns.append(word)
        return nouns
    except Exception as e:
        print(f"Error during MeCab parsing: {e}")
        return text  # Return original text on failure


def evaluate_metrics_threshold(df, retriever):
    """
    Evaluate retrieval metrics (Hit@K, MRR, Precision) with a given retriever.

    Args:
        df (pd.DataFrame): Evaluation dataset.
        retriever: Retriever object to perform document retrieval.

    Returns:
        tuple: Result dataframe, Hit@K, MRR@K, Average Precision.
    """
    result_df = df.copy()
    result_df['reference'] = ""
    
    for idx, row in tqdm(result_df.iterrows()):
        retrieved_docs = retriever.invoke(row['query'])
        if retrieved_docs: # If documents are retrieved
            references = [ref.page_content for ref in retrieved_docs]
            result_df.loc[idx, 'reference'] = str(references)
        else:  # If no documents are retrieved
            result_df.loc[idx, 'reference'] = ""

    # Metric 계산 초기화
    total_hits = 0  # Total number of hits
    total_reciprocal_rank = 0.0  # Sum of reciprocal ranks
    total_precision = 0.0  # Sum of precision
    valid_rows = 0  # Number of valid rows (rows with references)

    result_df[['hit', 'rank', 'precision']] = [False, 0, 0.0]

    for idx, row in tqdm(result_df.iterrows()):
        # Skip if reference is empty
        if not row['reference'] or row['reference'] == '[]':
            continue

        # Split keywords by commas
        keywords = [kw.strip() for kw in row['keyword'].split(',')]

        # List of retrieved documents
        references = eval(row['reference'])

        K = len(references)  # Number of retrieved documents (Top K)
        relevant_retrieved_docs = 0  # Number of relevant documents retrieved
        rank = 0  # Rank of the first relevant document
        found = False

        for i, doc in enumerate(references):
            # Remove whitespace from document
            doc_no_space = doc.replace(' ', '').replace('\n', '')
            doc_is_relevant = False
            for kw in keywords:
                # Remove whitespace from keywords
                kw_no_space = kw.replace(' ', '')
                # Check if keyword is in the document
                if kw_no_space in doc_no_space:
                    doc_is_relevant = True
                    if not found:
                        rank = i + 1  # Rank starts from 1
                        found = True
                    break  # Exit inner loop as keyword is found
            if doc_is_relevant:
                relevant_retrieved_docs += 1  # Increment count of relevant documents

        if rank > 0:
            result_df.loc[idx, 'hit'] = True
            result_df.loc[idx, 'rank'] = rank
            total_hits += 1  # Increment hit count
            reciprocal_rank = 1.0 / rank  # Calculate reciprocal rank
            total_reciprocal_rank += reciprocal_rank  # Add to total reciprocal rank

        # Calculate precision for current query
        precision = relevant_retrieved_docs / K if K > 0 else 0
        result_df.loc[idx, 'precision'] = precision
        total_precision += precision

        valid_rows += 1  # Increment count of valid rows

    # Calculate overall metrics
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
    """
    Evaluate retrieval metrics (Hit@K, MRR, Precision) using Jaccard similarity re-ranking.

    Args:
        df (pd.DataFrame): Evaluation dataset.
        retriever: Retriever object to perform document retrieval.
        topk (int): Number of top documents to re-rank and evaluate.

    Returns:
        tuple: Result dataframe, Hit@K, MRR@K, Average Precision.
    """
    result_df = df.copy()
    result_df['reference'] = ""
    
    for idx, row in tqdm(result_df.iterrows(), total=len(result_df)):
        # Retrieve topk_base documents using dense retriever
        retrieved_docs = retriever.invoke(row['query'])
        if retrieved_docs:  # If documents are retrieved
            # Re-rank documents using Jaccard similarity
            reranked_docs = jaccard_reranker(row['query'], retrieved_docs, topk=topk)
            references = [ref.page_content for ref in reranked_docs]
            result_df.loc[idx, 'reference'] = str(references)
        else:  # If no documents are retrieved
            result_df.loc[idx, 'reference'] = "[]"

    # Initialize metrics
    total_hits = 0  # Total number of hits
    total_reciprocal_rank = 0.0  # Sum of reciprocal ranks
    total_precision = 0.0  # Sum of precision
    valid_rows = 0  # Number of valid rows (rows with references)

    result_df[['hit', 'rank', 'precision']] = [False, 0, 0.0]

    for idx, row in tqdm(result_df.iterrows(), total=len(result_df)):
        # Skip if reference is empty
        if not row['reference'] or row['reference'] == '[]':
            continue

        # Split keywords by commas
        keywords = [kw.strip() for kw in row['keyword'].split(',')]

        # List of retrieved documents
        references = literal_eval(row['reference'])

        K = len(references)  # Number of retrieved documents (Top K)
        relevant_retrieved_docs = 0  # Number of relevant documents retrieved
        rank = 0  # Rank of the first relevant document
        found = False

        for i, doc in enumerate(references):
            # Remove whitespace from document
            doc_no_space = doc.replace(' ', '').replace('\n', '')
            doc_is_relevant = False
            for kw in keywords:
                # Remove whitespace from keywords
                kw_no_space = kw.replace(' ', '')
                # Check if keyword is in the document
                if kw_no_space in doc_no_space:
                    doc_is_relevant = True
                    if not found:
                        rank = i + 1  # Rank starts from 1
                        found = True
                    break  # Exit inner loop as keyword is found
            if doc_is_relevant:
                relevant_retrieved_docs += 1  # Increment count of relevant documents

        if rank > 0:
            result_df.loc[idx, 'hit'] = True
            result_df.loc[idx, 'rank'] = rank
            total_hits += 1  # Increment hit count
            reciprocal_rank = 1.0 / rank  # Calculate reciprocal rank
            total_reciprocal_rank += reciprocal_rank  # Add to total reciprocal rank

        # Calculate precision for current query
        precision = relevant_retrieved_docs / K if K > 0 else 0
        result_df.loc[idx, 'precision'] = precision
        total_precision += precision

        valid_rows += 1  # Increment count of valid rows

    # Calculate overall metrics
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
    """
    Re-rank retrieved documents based on Jaccard similarity.

    Args:
        query (str): Input query.
        retrieved_docs (list): Retrieved documents.
        topk (int): Number of top documents to return.
    
    Returns:
        list: Re-ranked documents.
    """
    tokenized_query = extract_nouns(query)
    context_scores = []
    for doc in retrieved_docs:
        tokenized_context = extract_nouns(doc.page_content)
        common_terms = set(tokenized_query).intersection(set(tokenized_context))
        score = len(common_terms) / len(set(tokenized_query).union(set(tokenized_context))) 
        context_scores.append(score)

    # Re-rank contexts based on similarity scores and return topk documents
    reranked_docs = [doc for _, doc in sorted(zip(context_scores, retrieved_docs), key=lambda x: x[0], reverse=True)][:topk]
    
    return reranked_docs


def process_row(row):
    """
    Process a single row from the evaluation dataset for retrieval.

    Args:
        row (pd.Series): Single row from the dataset.
    
    Returns:
        list: Retrieved documents.
    """
    problems = literal_eval(row['problems'])
    paragraph = row['paragraph']
    question = problems['question']
    choices = problems['choices']
    choices_str = " ".join(choices)
    
    query = prompt.format(paragraph=paragraph, question=question, choices=choices_str)
    
    # Conservative criterion: Exclude documents longer than 500 characters
    if len(str(paragraph)) > 500:
        return []
    
    # Retrieve documents using dense retriever
    docs = faiss_retriever.invoke(query)
    
    if docs:
        # Re-rank using Jaccard similarity
        reranked_docs = jaccard_reranker(query, docs, topk=topk)
        # Extract topk documents
        retrieved_docs = [doc.page_content for doc in reranked_docs]
        return retrieved_docs
    else:
        return []


# Load evaluation dataset
eval_set = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/rag_eval/external_knowledge_w_label_keyword.csv")
prompt = "{paragraph}\n{question}\n{choices}"
eval_set['query'] = ""
for idx, row in eval_set.iterrows():
    problems = literal_eval(row['problems'])
    question = problems['question']
    choices = problems['choices']
    choices_str = " ".join(choices)
    query = prompt.format(paragraph=row['paragraph'], question=question, choices=choices_str)
    eval_set.loc[idx, 'query'] = query

# Load and process RAG data
rag_folder = "/data/ephemeral/home/workspace/contest_baseline_code/data/rag"
rag_files = glob(f"{rag_folder}/*.csv")
rag_data_source = [pd.read_csv(file) for file in rag_files]
rag_data = pd.concat(rag_data_source, axis=0, ignore_index=True)
print(f"RAG Data Count: {rag_data.shape[0]}")

# Filter documents
rag_data = rag_data[rag_data.context.str.len() >= 25]
print(f"filtered (len > 25) RAG Data Count: {rag_data.shape[0]}")

# Chunk documents
loader = DataFrameLoader(rag_data, page_content_column='context')
documents = loader.load()

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

# Load vector store
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

    batch_size = 4
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


prompt = "{paragraph} {question} {choices}"

# Save Train Dataset
target_data = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/preprocessed/train_fix_khan_kor_v2_korean.csv")
    
target_data['reference'] = target_data.progress_apply(process_row, axis=1)

target_data.to_csv(f"/data/ephemeral/home/workspace/contest_baseline_code/data/preprocessed/train_rag_rerank{topk}_final.csv", index=False)

# Save Test Dataset
target_data = pd.read_csv("/data/ephemeral/home/workspace/contest_baseline_code/data/raw/test.csv")
target_data['reference'] = target_data.progress_apply(process_row, axis=1)

target_data.to_csv(f"/data/ephemeral/home/contest_baseline_code/data/preprocessed/test_rag_rerank{topk}_v2_list.csv", index=False)