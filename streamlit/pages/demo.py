import time
import pandas as pd
from PIL import Image
import streamlit as st
from ast import literal_eval

st.set_page_config(
     page_title='Generation for NLP - Korean SAT',
     layout="wide",
     initial_sidebar_state="expanded",
)

st.sidebar.page_link("home.py", label="🏠 Home")
st.sidebar.page_link("pages/architecture.py", label="🏢 Architecture")
st.sidebar.page_link("pages/demo.py", label="🤖 Demo")

KEYS = ["2025-korean-01", "2025-history-03", "2025-history-13"]
for key in KEYS:
    if key not in st.session_state:
        st.session_state[key] = False

ksat_df = pd.read_csv("assets/ksat_dataset.csv")

st.header("Lv.2 Generation for NLP - 2025학년도 수능 with RAG")
st.subheader("2025학년도 수능 국어, 사회 영역 문제 풀이")

def ksat_demo(ksat_id):
    target = ksat_df.loc[ksat_df["id"]==ksat_id]
    references = literal_eval(target["reference"].values[0])
    
    st.markdown(
        f"""
        <div style="background-color: #F1F1EF; padding: 10px; border-radius: 5px;">
            <br>
            <p style="font-size: 18px;"><b>정답</b></p>
            <p>{target["answer_true"].values[0]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(" ")
    if st.button("단일 모델 결과", key=ksat_id+"-only"):
        st.session_state[ksat_id] = True
    if st.session_state[ksat_id]:
        time.sleep(2)
        st.markdown(
            f"""
            <div style="background-color: #E7F3F8; padding: 10px; border-radius: 5px;">
                <p style="font-size: 18px;"><b>단일 모델 예측 결과</b></p>
                <p>{target["answer_pred"].values[0]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(" ")

    if ksat_id != KEYS[0]:
        if st.button("RAG 결과", key=ksat_id+"-rag"):
            time.sleep(3)

            st.markdown(
                f"""
                <div style="background-color: #FBF3DB; padding: 10px; border-radius: 5px;">
                    <p style="font-size: 18px;"><b>참고 문서</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            for i, reference in enumerate(references):
                st.markdown(
                    f"""
                    <div style="background-color: #FBF3DB; padding: 10px; border-radius: 5px;">
                        <p><b>{i+1}번째</b><br>{reference}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown(
                f"""
                <div style="background-color: #FBF3DB; padding: 10px; border-radius: 5px;">
                    <p style="font-size: 18px;"><b>RAG 예측 결과</b></p>
                    <p>{target["answer_rag"].values[0]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

score_tab, korean_tab, social_tab1, social_tab2 = st.tabs(["🎯 점수", "📖 국어", "📰 한국사 1", "📰 한국사 2"])
with score_tab:
    st.subheader("채점 결과")
    # score_df = pd.read_csv("assets/score.csv").set_index("과목")
    # st.dataframe(score_df, width=400)
    score_img = Image.open("assets/score.jpg").resize((1100, 350))
    st.image(score_img)

with korean_tab:
    st.subheader("국어")

    kr_col1, kr_col2 = st.columns(2)
    with kr_col1:
        img = Image.open(f"assets/{KEYS[0]}.png").resize((380, 900))
        st.image(img)
    
    with kr_col2:
        ksat_demo(KEYS[0])

with social_tab1:
    st.subheader("한국사 - 예시 1")

    so1_col1, so1_col2 = st.columns(2)
    with so1_col1:
        img =Image.open(f"assets/{KEYS[1]}.png").resize((450, 400))
        st.image(img)
    
    with so1_col2:
        ksat_demo(KEYS[1])

with social_tab2:
    st.subheader("한국사 - 예시 2")

    so2_col1, so2_col2 = st.columns(2)
    with so2_col1:
        img =Image.open(f"assets/{KEYS[2]}.png").resize((450, 400))
        st.image(img)
    
    with so2_col2:
        ksat_demo(KEYS[2])