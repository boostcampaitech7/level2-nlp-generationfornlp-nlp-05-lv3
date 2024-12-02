# 🏆 LV.2 NLP 프로젝트 : 수능형 문제 풀이 모델 생성

## ✏️ 대회 소개

| 특징     | 설명 |
|:------:| --- |
| 대회 주제 | 네이버 부스트캠프 AI Tech 7기 NLP Track의 Level 2 도메인 기초 대회 '수능형 문제 풀이 모델 생성'입니다. |
| 대회 설명 | AI 모델로 한국어 수능 국어 및 사회 과목의 문제를 풀어 대형 언어 모델들을 능가하는 성능을 목표로 하는 대회입니다. |
| 데이터 구성 | 데이터는 수능 국어·사회와 유사한 문제를 기반으로 KMMLU(한국사), MMMLU(고교 역사·경제·정치 등), KLUE MRC(경제·국제·사회 등) 데이터를 포함합니다. |
| 평가 지표 | 모델이 맞춘 문제 수를 전체 문제 수로 나눈 정확도(Accuracy)를 기준으로 합니다.|
| 결과물 | WrapUp Report |

## 🎖️ Leader Board

프로젝트 결과 Public 리더보드 1등, Private 리더보드 1등을 기록하였습니다.

### 🥇 Public Leader Board (1위)

(이미지)

### 🥇 Private Leader Board (1위)

(이미지)


## 👨‍💻 나야, 자, 연어팀 멤버
<div align='center'>
  
| 곽희준 [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=20 style="vertical-align:middle;" />](https://github.com/gwaksital) | 김정은 [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=20 style="vertical-align:middle;" />](https://github.com/wjddms4299) | 김진재 [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=20 style="vertical-align:middle;" />](https://github.com/jin-jae) | 오수현 [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=20 style="vertical-align:middle;" />](https://github.com/ocean010315) | 윤선웅 [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=20 style="vertical-align:middle;" />](https://github.com/ssunbear) | 정민지 [<img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width=20 style="vertical-align:middle;" />](https://github.com/minjijeong98) 
|:-:|:-:|:-:|:-:|:-:|:-:|
| ![곽희준](https://avatars.githubusercontent.com/u/80732503) | ![김정은](https://avatars.githubusercontent.com/u/121777522) | ![김진재](https://avatars.githubusercontent.com/u/97018331) | ![오수현](https://avatars.githubusercontent.com/u/91974779) | ![윤선웅](https://avatars.githubusercontent.com/u/117508164) | ![정민지](https://avatars.githubusercontent.com/u/162319450) |

</div>



## 👼 역할 분담

<div align='center'>

| 팀원 | 역할 |
| --- | --- |
| 곽희준 | 데이터 레이블링, EDA, 외부 데이터셋 탐색, GPT를 통한 데이터셋 증강 실험, 코드 리팩토링, LLM 학습 방법 설계, Fine-Tuning, RAG 파이프라인 구축, 최종 코드 정리 |
| 김정은 | 데이터셋 레이블링, 모델 탐색 및 Fine-Tuning, 데이터셋 크롤링 및 전처리(공무원 기출, khan), 데이터셋 품질 테스트, RAG 시스템 구축 및 실험, 프롬프트 엔지니어링 |
| 김진재 | 초기 팀 환경 구축 및 대시보드 제작, 데이터셋 레이블링, 데이터 탐색, RAG 데이터 전처리(번역), Retrieval 구축 및 실험 (Sparse) |
| 오수현 | 초기 베이스라인 코드 구축, 데이터셋 레이블링, 데모 페이지 제작 |
| 윤선웅 | 데이터셋 레이블링, 모델 탐색 및 Fine-Tuning, Unsloth 세팅, 프롬프트 엔지니어링, 데이터셋 크롤링(공무원 기출, khan), 데이터셋 품질 테스트, LoRA 튜닝, 앙상블 |
| 정민지 | 데이터셋 레이블링, 벡터스토어 데이터 크롤링 및 전처리 (OpenStax, Wikipedia, 우리역사넷), Retrieval 성능 평가 데이터셋 및 지표 구성, RAG 시스템 구축 및 실험 (Chunk size, Dense Retrieval, Reranking) |

</div>

## 🏃 프로젝트

### 🖥️ 프로젝트 개요

| 개요 | 설명 |
| --- | --- |
| 주제 | Generation for NLP - 수능형 문제 풀이 모델 생성 |
| 구조 | LLM Fine-Tuned Foundation Model + RAG |
| 평가 지표 | Accuracy = correct / total |
| 개발 환경 | `GPU` : Tesla V100 Server 4대, `IDE` : Vscode, Jupyter Notebook |
| 협업 환경 | Jira&Confluence(진행 상황 공유), Github(코드 및 데이터 공유), Zoom&Slack(실시간 소통) |

### 📅 프로젝트 타임라인

- 프로젝트는 2024-11-11 ~ 2024-11-28까지 진행되었습니다.

(이미지)

### 🕵️ 프로젝트 진행

- 프로젝트를 진행하며 단계별로 실험하여 적용한 내용들은 아래와 같습니다.

| 프로세스 | 설명 |
| --- | --- |
| 데이터 | EDA, Fine-Tuning 데이터셋 구성 (데이터 품질 개선, 데이터 증강), RAG 데이터 구성 (벡터 스토어 데이터 수집 및 전처리, chunking) |
| 모델링 | 모델 선정 및 튜닝, LoRA 튜닝, 프롬프트 튜닝 |
| RAG | Vector Store 구축, Retriever 평가용 데이터셋 구축, Retriever 파라미터 설정, RAFT(Retrieval Augmented Fine-Tuning) |
| 앙상블 | Weighted Voting Ensemble |

### 🤖 Ensemble

정제, 증강을 다양하게 적용한 데이터셋과 LoRA 튜닝을 통해 `itsmenlp/unsloth_qwen_2.5_32B_bnb_4bit_finetuned`로 추론한 output의 accuracy Top 5로 weighted voting ensemble을 진행한 결과, 최종 Public Accuracy **0.8341**을 달성했습니다.

| Output | Accuracy | Weight |
| --- | --- | --- |
| Top 5 | 0.8180 | 0.1 |
| Top 4 | 0.8180 | 0.1 |
| Top 3 | 0.8203 | 0.2 |
| Top 2 | 0.8272 | 0.2 |
| Top 1 | 0.8295 | 0.4 |

### 📃 KSAT Results

본 프로젝트에서 개발한 sLLM을 활용한 2025학년도 수능의 국어(화법과 작문), 한국사, 사회 탐구 영역 풀이 결과입니다.

(이미지)


## 🎥 데모 영상

https://github.com/user-attachments/assets/4448f058-6571-4037-9fb9-dfd8f86d5291

## 📁 프로젝트 구조

프로젝트 폴더 구조는 다음과 같습니다.
