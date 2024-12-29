from PIL import Image
import streamlit as st

st.set_page_config(
     page_title='Generation for NLP - Korean SAT',
     layout="wide",
     initial_sidebar_state="expanded",
)

st.sidebar.page_link("home.py", label="🏠 Home")
st.sidebar.page_link("pages/architecture.py", label="🏢 Architecture")
st.sidebar.page_link("pages/demo.py", label="🤖 Demo")

st.header("Lv.2 Generation for NLP - 2025학년도 수능 with RAG")

st.subheader("개요")
img = Image.open("assets/tutorial.png").resize((900, 300))
st.image(img)

st.markdown("##### 목표")
st.markdown("작은 규모의 모델을 사용하여 수능 문제를 정확하게 풀 수 있는 AI 모델 개발")
            
st.markdown("##### 평가 지표")
st.markdown("Accuracy = 모델이 맞춘 문제의 수 / 전체 문제의 수")

st.markdown("##### 데이터")
st.markdown("""**입력:** 수능 국어와 사회 과목의 지문 형태, `id`, `paragraph`, `problems(question, choices, answer)`, `question_plus`  
            **출력:** 주어진 선택지(choices) 중에서 정답에 해당하는 번호 출력""")
            
st.markdown("##### 의의")
st.markdown("""1. 수능형 문제를 정확히 풀어냄으로써 자연어 이해는 물론 모델의 맥락 파악, 논리적 추론, 정보의 종합 능력을 보다 복합적으로 평가하고 향상시킨다.  
2. 보다 작은 모델로 본 프로젝트를 수행하며 자원의 효율성, 실용성을 달성하며 확장 가능성을 확인할 수 있다.""")
