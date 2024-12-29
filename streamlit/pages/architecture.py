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

st.subheader("구조도")
# st.markdown("""**Retriever:** Sparse & Dense Retriever → Re-Ranker  
#             **Reader:** `itsmenlp/unsloth_qwen_2.5_32B_bnb_4bit_finetuned`""")
arc_img = Image.open("assets/architecture.png")
st.image(arc_img)
